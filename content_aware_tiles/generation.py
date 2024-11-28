import torch
from torch import Tensor
from einops import rearrange, einsum
import numpy as np
from typing import Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

def unslide_wrapping(z, off_c, off_r):
    w, h = z.shape[-2:]
    off_c = off_c % w
    off_r = off_r % h
    z = torch.cat([z, z], -1)
    z = torch.cat([z, z], -2)
    z = z[..., off_c:w+off_c, off_r:h+off_r]
    return z


def slide_wrapping(z, off_c, off_r):
    w, h = z.shape[-2:]
    off_c = off_c % w
    off_r = off_r % h
    z = torch.cat([z, z], -1)
    z = torch.cat([z, z], -2)
    z = z[..., w-off_c:2*w-off_c, h-off_r:2*h-off_r]
    return z

def wang_kernel(n, m):
    return [
        [[0, 0], [0, 0    ], [0  , 0]],
        [[0, 0], [1, n    ], [n*m, 0]],
        [[0, 0], [0, n*m*n], [0  , 0]],
    ]

def cross_kernel(n, m):
    assert n == m # not handled yet
    return [
        [[0, 0], [n,     0  ], [0, 0]],
        [[0, 1], [n*n*n, n*n], [0, 0]],
        [[0, 0], [0,     0  ], [0, 0]],
    ]


def corner_kernel(n):
    return [
        [[0], [0  ], [0    ]],
        [[0], [1  ], [n    ]],
        [[0], [n*n], [n*n*n]],
    ]

def tile_kernel(kind: str, colors: int):
    if kind.startswith('wang'):
        kernel = wang_kernel(colors, colors)
    elif kind.startswith('cross'):
        kernel = cross_kernel(colors, colors)
    elif kind.startswith('corner'):
        kernel = corner_kernel(colors)
    elif kind.startswith('dual'):
        kernel = torch.cat([
            tile_kernel('wang', colors),
            tile_kernel('cross', colors),
        ], 0)
    else:
        assert False
    if isinstance(kernel, list):
        kernel = torch.tensor(kernel, dtype=torch.int64).permute(2, 0, 1).unsqueeze(0)
    #print('kernel =', kernel.shape)
    return kernel

def convolve_tile_indicies(field: Tensor, colors: int, kind: str) -> Tensor:
    if kind == 'self':
        return field
    kernel = tile_kernel(kind, colors)
    field = torch.nn.functional.pad(field, (1, 1, 1, 1), mode='circular')
    field = torch.nn.functional.conv2d(field, kernel)
    batches, sets, iw, ih = field.shape
    for i in range(sets):
        field[:, i, ...] += colors ** 4 * i
    return field

def random_tile_indicies(batch: int, height: int, width: int, colors: int, kind: str, return_field=False) -> Tensor|Tuple[Tensor,Tensor]:
    """
    Generate random Wang indices for various tiling schemes.

    This function creates random indices for different tiling types, including
    self-tiling, Wang tiles, cross tiles, corner tiles, and dual tiles.

    Args:
        batch (int): Number of batches to generate.
        height (int): Height of the index field.
        width (int): Width of the index field.
        colors (int): Number of colors or edge conditions.
        kind (str): Type of tiling scheme ('self', 'wang', 'cross', 'corner', or 'dual').
        return_field (bool): If True, return both the random field and convolved indices.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Convolved indices, or both
        random field and convolved indices if return_field is True.

    Raises:
        AssertionError: If an invalid tiling kind is specified.
    """
    
    if kind.startswith('self'):
        features = 1  
    elif kind.startswith('wang'):
        features = 2
    elif kind.startswith('cross'):
        features = 2
    elif kind.startswith('corner'):
        features = 1
    elif kind.startswith('dual'):
        features = 2
    else:
        assert False
    field = torch.randint(0, colors, (batch, features, height, width), dtype=torch.int64)
    convolved = convolve_tile_indicies(field, colors, kind)
    if return_field:
        return field, convolved
    else:
        return convolved

def place_tiles(tiles: Tensor, colors: int, kind: str, indices:Tensor|np.ndarray|Tuple[int,int]|Tuple[int,int,int]=(8, 8)):
    """
    Create a tiling from given tiles based on specified indices or generate random indices.

    This function supports various tiling schemes and can handle both predetermined
    and randomly generated indices for tile placement.

    Args:
        tiles (torch.Tensor): Tensor containing tile data.
        colors (int): Number of colors or edge conditions.
        kind (str): Type of tiling scheme.
        indices (tuple or torch.Tensor): Predetermined indices or dimensions for random generation.

    Returns:
        torch.Tensor: The resulting tiled image.

    Raises:
        ValueError: If the provided indices are in an invalid format.
        AssertionError: If an unsupported tiling kind is specified.
    """

    assert isinstance(kind, str)
    if isinstance(indices, torch.Tensor):
        has_batch = len(indices.shape) == 4
        if not has_batch:
            indices = indices[None]
    elif isinstance(indices, np.ndarray):
        has_batch = len(indices.shape) == 4
        indices = torch.tensor(indices, device=tiles.device, dtype=torch.int64)
        if not has_batch:
            indices = indices[None]
    else:
        if len(indices) == 2:
            has_batch = False
            h, w = indices
            b = 1
        elif len(indices) == 3:
            has_batch = True
            b, h, w = indices
        else:
            raise ValueError(f'can not interpret indices with shape {indices}')
        if kind == 'self':
            indices = random_tile_indicies(b, h, w, tiles.shape[0], kind)
        else:
            indices = random_tile_indicies(b, h, w, colors, kind)
    assert isinstance(indices, torch.Tensor)

    y = tiles[indices]
    b, l, iw, ih, c, tw, th = y.shape
    y = rearrange(y, 'b l iw ih c tw th -> l b c (iw tw) (ih th)')
    
    if l == 1:
        tiling = y[0]
    elif kind.startswith('dual'):
        assert l == 2
        # offset the cross tiles to lie on each corner
        y[1] = slide_wrapping(y[1], -tw//2, -th//2)
        # now composite the interior/cross tiles with a diamond-shaped mask
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, steps=tw, dtype=torch.float32),
            torch.linspace(-1, 1, steps=th, dtype=torch.float32),
            indexing='ij',
        )
        mask = (torch.abs(i) + torch.abs(j)) < 1
        mask = mask.reshape(1, 1, tw, th).tile(1, 1, iw, ih).to(y.device)
        tiling = torch.where(mask, y[0], y[1])
    else:
        raise AssertionError(f"Unsupported tiling kind: {kind}")

    if not has_batch:
        tiling = tiling[0]
    return tiling

@dataclass
class Chunked:
    inner: Any
    mode: str

def split_chunks(x: Any, chunk_size) -> Any:
    if isinstance(x, torch.Tensor) and len(x.shape) > 0:
        return Chunked(inner=x.chunk(chunk_size), mode='tensor')
    elif isinstance(x, (tuple, list)):
        return Chunked(inner=type(x)(split_chunks(item, chunk_size) for item in x), mode='list')
    elif isinstance(x, dict):
        return Chunked(inner={k: split_chunks(v, chunk_size) for k, v in x.items()}, mode='dict')
    else:
        return x

def select_chunks(x: Any, i: int) -> Any:
    if not isinstance(x, Chunked):
        return x
    elif x.mode == 'tensor':
        return x.inner[i]
    elif x.mode == 'list':
        return type(x.inner)(select_chunks(v, i) for v in x.inner)
    elif x.mode == 'dict':
        return {k: select_chunks(v, i) for k, v in x.inner.items()}
    else:
        raise NotImplementedError(f'chunking mode {x.mode}')

def merge_chunks(chunks: list[Any]) -> Any:
    rep = chunks[0]
    if rep is None:
        return None
    elif isinstance(rep, torch.Tensor):
        return torch.cat(chunks, 0)
    elif isinstance(rep, (tuple, list)):
        merged = []
        for i in range(len(rep)):
            merged.append(merge_chunks([c[i] for c in chunks]))
        return type(rep)(merged)
    elif isinstance(rep, dict):
        merged = {}
        for k in rep.keys():
            merged[k] = merge_chunks([c[k] for c in chunks])
        return type(rep)(merged)
    elif type(rep).__name__ == 'DiagonalGaussianDistribution':
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        params = merge_chunks([d.parameters for d in chunks])
        return DiagonalGaussianDistribution(params)
    else:
        raise NotImplementedError(f'merging {type(chunks[0])}')

def call_chunked(func, *args, chunk_size=8, **kwargs):
    split_args = split_chunks(args, chunk_size)
    split_kwargs = split_chunks(kwargs, chunk_size)

    output_list = []
    i = 0
    while True:
        try:
            this_args = select_chunks(split_args, i)
            this_kwargs = select_chunks(split_kwargs, i)
        except IndexError:
            break
        output = func(*this_args, **this_kwargs)
        output_list.append(output)
        i += 1

    return merge_chunks(output_list)

def triangle_quadrant(side: str, size: int):
    # Create a meshgrid
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    y = y + 0.5

    if side == 'left':
        triangle = (x <= y) * (x < size - y)
    elif side == 'right':
        triangle = (x >= y) * (x > size - y)
    elif side == 'top':
        triangle = (x > y) * (x <= size - y)
    elif side == 'bottom':
        triangle = (x < y) * (x >= size - y)
    else:
        raise ValueError("Invalid side argument. Choose from 'left', 'right', 'top', or 'bottom'.")

    tensor = triangle.unsqueeze(0)
    return tensor

for size in [12, 13, 512]:
    assert torch.allclose(
        triangle_quadrant('left', size).to(torch.float32) + triangle_quadrant('right', size).to(torch.float32) + triangle_quadrant('top', size).to(torch.float32) + triangle_quadrant('bottom', size).to(torch.float32),
        torch.ones([1, size, size]),
    )

def get_packing(colors: int, kind: str):
    from .packing import create_packing
    packed_colors = torch.tensor(create_packing(colors), dtype=torch.int64)[None, :, :, [1, 0]]
    if colors == 3:
        assert torch.allclose(packed_colors, torch.tensor([int(i) for i in """
            0 0 0 2 1 2 1 1 0 2 2 0 1 1 2 1 2 0  
            0 0 1 0 1 2 0 2 2 1 1 2 2 0 2 1 0 1  
            1 1 1 0 0 0 2 2 1 2 2 1 2 2 0 0 0 1  
            1 1 0 1 2 0 1 0 2 2 2 2 0 1 0 2 1 0  
            0 0 2 1 1 1 2 0 2 0 0 2 0 2 1 1 1 2  
            2 2 1 0 2 1 2 1 0 0 0 0 1 2 1 2 0 1  
            1 1 2 2 2 0 0 1 0 1 1 0 1 0 0 2 2 2  
            2 2 2 1 0 2 0 0 1 1 1 1 0 0 2 0 1 2  
            2 2 0 2 0 1 1 2 1 0 0 1 2 1 1 0 2 0
        """.split()], dtype=torch.int64).reshape(1, 9, 9, 2))
    packed_colors = rearrange(packed_colors, 'b h w l -> b l h w')

    if kind == 'wang':
        packed_edge_indicies = convolve_tile_indicies(packed_colors, colors, 'wang')
        packed_tile_counts = torch.zeros(colors**4, dtype=torch.int64)
        packed_tile_counts[packed_edge_indicies] += 1
        assert torch.all(packed_tile_counts == 1)

        return packed_edge_indicies[0]
    elif kind == 'dual':
        packed_dual_indicies = convolve_tile_indicies(packed_colors, colors, 'dual')
        packed_tile_counts = torch.zeros(colors**4*2, dtype=torch.int64)
        packed_tile_counts[packed_dual_indicies] += 1
        assert torch.all(packed_tile_counts == 1)

        return packed_dual_indicies[0]
    else:
        raise ValueError(f'can not make paking for {kind} tiles')

def unpack_tiles(packed: torch.Tensor, colors: int, kind: str, size, assert_roundtrip=False):
    def unpack(x):
        return rearrange(x, 'c (tw w) (th h) -> (tw th) c w h', w=size, h=size)

    if kind == 'wang':
        tiles = unpack(packed)
        perm = get_packing(colors, kind)
    elif kind == 'dual':
        interior_tiles = unpack(packed)
        cross_tiles = unpack(slide_wrapping(packed, size//2, size//2))
        tiles = torch.cat([interior_tiles, cross_tiles], 0)
        perm = get_packing(colors, kind)
    elif kind == 'self':
        return unpack(packed)
    else:
        raise NotImplementedError(f'unpacking for {kind} tiles')

    unpermuted = tiles.clone()
    unpermuted[perm.reshape(-1)] = tiles

    if assert_roundtrip and kind != 'self':
        assert torch.allclose(place_tiles(unpermuted, 3, kind, perm), packed)
    
    return unpermuted

def pack_tiles(unpacked: torch.Tensor, colors: int, kind: str):
    if kind == 'wang' or kind == 'dual':
        perm = get_packing(colors, kind)
    elif kind == 'self':
        c = int(np.ceil(np.sqrt(unpacked.shape[0])))
        perm = torch.arange(c**2).reshape(1, c, c) % unpacked.shape[0]
    else:
        raise NotImplementedError()
        
    return place_tiles(unpacked, 3, kind, perm)

textile_model = None

@torch.no_grad()
def select_candidates(tiles: Tensor, source: Tensor, candidates: int, keep: int, metric: str) -> Tensor:
    """
    Select the best candidate tiles based on a specified metric.

    This function evaluates multiple candidate tiles using a given metric and selects
    the best ones to keep.

    Args:
        tiles (torch.Tensor): A tensor of candidate tiles with shape (candidates * tiles, channels, height, width).
        source (torch.Tensor): The source image tensor used for comparison in some metrics.
        candidates (int): The number of candidate tiles per position.
        keep (int): The number of best tiles to keep per position.
        metric (str): The metric to use for evaluating tiles. Options are 'sifid' or 'textile'.

    Returns:
        tiles (torch.Tensor): A tensor of selected tiles with shape (keep * tiles, channels, height, width).

    Raises:
        NotImplementedError: If an unsupported metric is specified.

    Note:
        - The function uses no gradients (@torch.no_grad()).
        - For the 'textile' metric, it uses a global 'textile_model' which is lazy-loaded.
        - The 'sifid' metric requires the source image for comparison.
    """

    assert keep <= candidates

    if keep == candidates:
        return tiles

    tiles = rearrange(tiles, '(c t) k h w -> t c k h w', c=candidates)

    if metric == 'sifid':
        from .sifid import sifid

        score = sifid(source)
    elif metric == 'textile':
        from textile import Textile

        global textile_model

        if textile_model is None:
            textile_model = Textile().eval().to(tiles.device)

        score = lambda x: -call_chunked(textile_model, x).item() # type: ignore
    else:
        raise NotImplementedError(metric)

    all_scored = []
    for i in range(tiles.shape[0]):
        scored = []
        for j in range(tiles.shape[1]):
            tile = tiles[i, j:j+1, :, :, :]
            scored.append((score(tile), tile))
        scored.sort(key=lambda pair: pair[0])
        all_scored.append(scored)

    selected_tiles = []
    for j in range(keep):
        for scored in all_scored:
                selected_tiles.append(scored[j][1])

    return torch.cat(selected_tiles, 0)

def random_cuts(source: torch.Tensor, resolution: int, colors: int, mode: str) -> torch.Tensor:
    batch, _, w, h = source.shape
    cuts = []
    single_off_c = 0 if w == resolution else torch.randint(w - resolution, []).item()
    single_off_r = 0 if h == resolution else torch.randint(h - resolution, []).item()
    source = torch.nn.functional.pad(source, (resolution//2, resolution//2, resolution//2, resolution//2), mode='circular', value=None)
    for i in range(colors * 2):
        if mode == 'ortho constrained' and i < colors:
            off_c = single_off_c
        else:
            off_c = 0 if w == resolution else torch.randint(w - resolution, []).item()
        if mode == 'ortho constrained' and i >= colors:
            off_r = single_off_r
        else:
            off_r = 0 if h == resolution else torch.randint(h - resolution, []).item()
        b = torch.randint(batch, [])
        cuts.append(source[b, :, off_c:, off_r:][..., :resolution*2, :resolution*2])
    cuts = torch.stack(cuts, 0)
    return cuts

def diamond_inpaint_mask(resolution: int, source: torch.Tensor, diameter: float=2):
    tw = int(resolution*diameter)
    th = int(resolution*diameter)
    i, j = torch.meshgrid(
        torch.linspace(-diameter, diameter, steps=tw, dtype=torch.float32),
        torch.linspace(-diameter, diameter, steps=th, dtype=torch.float32),
        indexing='ij',
    )
    mask = ((torch.abs(i) + torch.abs(j)) < 1).to(source.dtype)
    return mask.reshape(1, tw, th)


def inpainting_from_boundaries(source: torch.Tensor, colors: int, boundary_kind: str, source_kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate exterior boundary conditions for content-aware tile generation.

    This function supports various tiling schemes including self-tiling, Wang tiles,
    and the novel Dual Wang tiles introduced in the paper. It handles the intricate
    process of extracting and assembling boundary conditions from source textures
    or previously generated tiles, crucial for maintaining seamless tileability
    while promoting diversity in the synthesized textures.

    Args:
        source (torch.Tensor): The source image or tiles to extract boundaries from.
        colors (int): The number of colors for Wang tiles or edge conditions.
        boundary_kind (str): The type of tiling scheme ('wang', 'self', or 'dual').
        source_kind (str): The format of the source data ('cuts', 'wang', 'uncropped_wang', or 'dual').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - tile_outsides: Tensor containing the exterior boundary conditions for each tile.
            - outsides_mask: Mask indicating the regions to be inpainted.
    """

    if (boundary_kind == 'wang' or boundary_kind == 'self') and source_kind == 'cuts':
        tile_size = source.shape[-1]//2
        crop = tile_size//2

        tile_outsides = []
        for index in range(colors**4):
            indicies = index // (colors ** np.arange(4)) % colors

            if source.shape[0] == colors:
                pass
            elif source.shape[0] == 2 * colors:
                indicies[1::2] += colors
            else:
                assert False

            l, t, r, b = source[indicies]
            l = slide_wrapping(l, 0, -tile_size//2) * triangle_quadrant("left", tile_size*2)
            r = slide_wrapping(r, 0, tile_size//2) * triangle_quadrant("right", tile_size*2)
            t = slide_wrapping(t, -tile_size//2, 0) * triangle_quadrant("top", tile_size*2)
            b = slide_wrapping(b, tile_size//2, 0) * triangle_quadrant("bottom", tile_size*2)
            tile_outsides.append(l + t + r +b)

        tile_outsides = torch.stack(tile_outsides, 0)

        d = {'dtype': tile_outsides.dtype, 'device': tile_outsides.device}
        z = torch.zeros(1, tile_size, tile_size, **d)
        o = torch.ones(1, tile_size, tile_size, **d)
        outsides_mask = torch.cat([
            torch.cat([o, z, o], -1),
            torch.cat([z, o, z], -1),
            torch.cat([o, z, o], -1),
        ], -2)
        outsides_mask = outsides_mask[:, crop:-crop, crop:-crop]
    elif boundary_kind == 'dual' and source_kind == 'wang':
        tile_size = source.shape[-1]
        crop = tile_size//2

        small_outsides = {}
        def get_small_outsides(c, i, j, x):
            if (c, i, j) not in small_outsides:
                small_outsides[(c, i, j)] = x
            return small_outsides[(c, i, j)]

        tile_outsides = []

        for index in range(colors**4):
            indicies = [(index // colors**i) % colors for i in range(4)]

            field = torch.randint(0, colors, (2, 3, 3), dtype=torch.int64)
            l, t, r, b = indicies
            field[0, 1, 1] = l
            field[1, 1, 1] = t
            field[0, 1, 2] = r
            field[1, 2, 1] = b
            field = convolve_tile_indicies(field.unsqueeze(0), colors, 'wang')
            combined = place_tiles(source, colors, 'wang', field)
            small_view = combined[:, :, tile_size:-tile_size, tile_size:-tile_size]
            small = small_view.clone()

            lt = get_small_outsides(0, l, t, small[..., :crop, :crop])
            rt = get_small_outsides(1, r, t, small[..., :crop, crop:])
            lb = get_small_outsides(2, l, b, small[..., crop:, :crop])
            rb = get_small_outsides(3, r, b, small[..., crop:, crop:])

            small_view.copy_(torch.cat([
                torch.cat([lt, rt], 3),
                torch.cat([lb, rb], 3),
            ], 2))

            combined = combined[..., crop:-crop, crop:-crop]
            tile_outsides.append(combined)

        tile_outsides = torch.cat(tile_outsides)
        outsides_mask = diamond_inpaint_mask(tile_size, source)
    elif boundary_kind == 'dual' and source_kind == 'uncropped_wang':
        full = source.shape[-1]
        half = full//2
        crop = half//2
    
        quarters = {}
        def get_quarter(c, i, j, x):
            if (c, i, j) not in quarters:
                quarters[(c, i, j)] = x
            return quarters[(c, i, j)]

        tile_outsides = []

        for index in range(colors**4):
            indicies = [(index // colors**i) % colors for i in range(4)]
            l, t, r, b = indicies
            combined = source[index:index+1].clone()
        
            lt = get_quarter(0, l, t, combined[..., :half, :half])
            rt = get_quarter(1, r, t, combined[..., :half, half:])
            lb = get_quarter(2, l, b, combined[..., half:, :half])
            rb = get_quarter(3, r, b, combined[..., half:, half:])

            combined = torch.cat([
                torch.cat([lt, rt], 3),
                torch.cat([lb, rb], 3),
            ], 2)

            tile_outsides.append(combined)
    
        tile_outsides = torch.cat(tile_outsides)
        outsides_mask = diamond_inpaint_mask(full, source)
    elif boundary_kind == 'wang' and source_kind == 'wang':
        tile_size = source.shape[-1]
        crop = tile_size//2

        tile_outsides = []
        for index in range(colors**4):
            indicies = [(index // colors**i) % colors for i in range(4)]

            field = torch.randint(0, colors, (2, 3, 3), dtype=torch.int64)
            l, t, r, b = indicies
            field[0, 1, 1] = l
            field[1, 1, 1] = t
            field[0, 1, 2] = r
            field[1, 2, 1] = b
            field = convolve_tile_indicies(field.unsqueeze(0), colors, 'wang')
            combined = place_tiles(source, colors, 'wang', field)
            combined = combined[..., crop:-crop, crop:-crop]
            tile_outsides.append(combined)
    
        tile_outsides = torch.cat(tile_outsides)
        d = {'dtype': tile_outsides.dtype, 'device': tile_outsides.device}
        z = torch.zeros(1, tile_size, tile_size, **d)
        o = torch.ones(1, tile_size, tile_size, **d)
        outsides_mask = torch.cat([
            torch.cat([z, z, z], -1),
            torch.cat([z, o, z], -1),
            torch.cat([z, z, z], -1),
        ], -2)
        outsides_mask = outsides_mask[:, crop:-crop, crop:-crop]
    elif boundary_kind == 'dual' and ((source_kind == 'dual' and source.shape[0] == colors**4) or source_kind == 'wang'):
        tile_size = source.shape[-1]
        tile_outsides = []

        for index in range(colors**4):
            indicies = [(index // colors**i) % colors for i in range(4)]

            field = torch.randint(0, colors, (2, 3, 3), dtype=torch.int64)
            l, t, r, b = indicies
            field[1, 1, 0] = l
            field[0, 0, 1] = t
            field[1, 1, 1] = r
            field[0, 1, 1] = b
            # we use 'wang' here because we only care about using the interior tiles from the set
            # (in fact, only the interior tiles exist at this point)
            field = convolve_tile_indicies(field.unsqueeze(0), colors, 'wang')[..., :2, :2]
            combined = place_tiles(source, colors, 'wang', field)
            tile_outsides.append(combined)
   
        tile_outsides = torch.cat(tile_outsides)
        outsides_mask = diamond_inpaint_mask(tile_size, source)
    elif boundary_kind == 'dual' and source_kind == 'dual':
        tile_size = source.shape[-1]
        tile_outsides = []
        crop = tile_size//2

        for index in range(colors**4):
            indicies = [(index // colors**i) % colors for i in range(4)]

            field = torch.randint(0, colors, (2, 3, 3), dtype=torch.int64)
            l, t, r, b = indicies
            field[0, 1, 1] = l
            field[1, 1, 1] = t
            field[0, 1, 2] = r
            field[1, 2, 1] = b
            field = convolve_tile_indicies(field.unsqueeze(0), colors, 'dual')
            combined = place_tiles(source, colors, 'dual', field)
            combined = combined[..., crop:-crop, crop:-crop]
            tile_outsides.append(combined)

        for index in range(colors**4):
            indicies = [(index // colors**i) % colors for i in range(4)]

            field = torch.randint(0, colors, (2, 3, 3), dtype=torch.int64)
            l, t, r, b = indicies
            field[1, 1, 0] = l
            field[0, 0, 1] = t
            field[1, 1, 1] = r
            field[0, 1, 1] = b
            field = convolve_tile_indicies(field.unsqueeze(0), colors, 'dual')
            combined = place_tiles(source, colors, 'dual', field)
            combined = combined[..., 0:-tile_size, 0:-tile_size]
            tile_outsides.append(combined)
   
        tile_outsides = torch.cat(tile_outsides)
        outsides_mask = diamond_inpaint_mask(tile_size, source)
    else:
        raise NotImplementedError(f'can not make {boundary_kind} boundaries from {source_kind}')

    return (tile_outsides, outsides_mask)
