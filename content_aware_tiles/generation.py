import torch
from torch import Tensor
from einops import rearrange, einsum
import numpy as np
from typing import Tuple, Any
from collections import defaultdict

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

def wang_edge_kernel(n, m):
    return [
        [[0, 0], [0, 0    ], [0  , 0]],
        [[0, 0], [1, n    ], [n*m, 0]],
        [[0, 0], [0, n*m*n], [0  , 0]],
    ]

def wang_cross_kernel(n, m):
    assert n == m # not handled yet
    return [
        [[0, 0], [n,     0  ], [0, 0]],
        [[0, 1], [n*n*n, n*n], [0, 0]],
        [[0, 0], [0,     0  ], [0, 0]],
    ]


def wang_corner_kernel(n):
    return [
        [[0], [0  ], [0    ]],
        [[0], [1  ], [n    ]],
        [[0], [n*n], [n*n*n]],
    ]

def wang_kernel(kind: str, colors: int):
    if kind.startswith('wang'):
        kernel = wang_edge_kernel(colors, colors)
    elif kind.startswith('cross'):
        kernel = wang_cross_kernel(colors, colors)
    elif kind.startswith('corner'):
        kernel = wang_corner_kernel(colors)
    elif kind.startswith('dual'):
        kernel = torch.cat([
            wang_kernel('wang', colors),
            wang_kernel('cross', colors),
        ], 0)
    else:
        assert False
    if isinstance(kernel, list):
        kernel = torch.tensor(kernel, dtype=torch.int64).permute(2, 0, 1).unsqueeze(0)
    #print('kernel =', kernel.shape)
    return kernel

def convolve_wang_indicies(field: Tensor, colors: int, kind: str):
    if kind == 'self':
        return field
    kernel = wang_kernel(kind, colors)
    field = torch.nn.functional.pad(field, (1, 1, 1, 1), mode='circular')
    field = torch.nn.functional.conv2d(field, kernel)
    batches, sets, iw, ih = field.shape
    for i in range(sets):
        field[:, i, ...] += colors ** 4 * i
    return field

def random_wang_indicies(batch: int, height: int, width: int, colors: int, kind: str, return_field=False):
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
    convolved = convolve_wang_indicies(field, colors, kind)
    if return_field:
        return field, convolved
    else:
        return convolved

def index_simple_patches(i: Tensor, x: Tensor, kind: str):
    y = x[i]
    b, l, iw, ih, c, tw, th = y.shape
    y = rearrange(y, 'b l iw ih c tw th -> l b c iw tw ih th')
    y = y.reshape(l, b, c, iw*tw, ih*th)
    if kind.startswith('dual'):
        for layer in range(l):
            if layer % 2 == 1:
                y[layer] = slide_wrapping(y[layer], -tw//2, -th//2)
    return y

def unindex_simple_patches(i: Tensor, y: Tensor, kind: str, outputs=None, include_edge=True):
    l, b, c, w, h = y.shape
    ib, il, iw, ih = i.shape
    th = w // iw
    tw = h // ih
    assert ib == b
    assert il == l

    y = y.clone()
    if kind.startswith('dual'):
        for layer in range(l):
            if layer % 2 == 1:
                y[layer] = unslide_wrapping(y[layer], -tw//2, -th//2)

    if outputs is None:
        outputs = defaultdict(lambda: [])

    for batch in range(b):
        for layer in range(l):
            for xpos in range(iw):
                for ypos in range(ih):
                    if xpos == 0 and not include_edge:
                        continue
                    if xpos == iw-1 and not include_edge:
                        continue
                    if ypos == 0 and not include_edge:
                        continue
                    if ypos == ih-1 and not include_edge:
                        continue
                    index = i[batch, layer, xpos, ypos].item()
                    patch = y[layer, batch, :, xpos*tw:, ypos*th:][..., :tw, :th]
                    outputs[index].append(patch)
                
    return outputs

def blend_patch_layers(inds: Tensor, x: Tensor, kind: str):
    layers, b, c, w, h = x.shape
    ib, il, iw, ih = inds.shape
    th = w // iw
    tw = h // ih
    assert ib == b
    assert il == layers

    if layers == 1:
        return x[0]
    elif kind.startswith('dual'):
        assert layers == 2
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, steps=tw, dtype=torch.float32),
            torch.linspace(-1, 1, steps=th, dtype=torch.float32),
            indexing='ij',
        )
        mask = (torch.abs(i) + torch.abs(j)) < 1
        mask = mask.reshape(1, 1, tw, th).tile(1, 1, iw, ih).to(x.device)
        return torch.where(mask, x[0], x[1])
    else:
        assert False

def merge_chunks(chunks: list[Any]):
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
        NotImplemented(f'merging {type(chunks[0])}')    

def call_chunked(func, *args, chunk_size=8, **kwargs):
    new_args = []
    new_kwargs = []
    num_chunks = None
    for x in args:
        if isinstance(x, torch.Tensor) and len(x.shape) > 0:
            if num_chunks is None:
                while x.shape[0] % chunk_size != 0:
                    chunk_size -= 1
                num_chunks = x.shape[0] // chunk_size
            new_args.append((x.chunk(num_chunks), True))
        else:
            new_args.append((x, False))
    for k, x in kwargs.items():
        if isinstance(x, torch.Tensor) and len(x.shape) > 0:
            if num_chunks is None:
                while x.shape[0] % chunk_size != 0:
                    chunk_size -= 1
                num_chunks = x.shape[0] // chunk_size
            new_kwargs.append((k, x.chunk(num_chunks), True))
        else:
            new_kwargs.append((k, x, False))
    output_list = []
    assert num_chunks is not None
    for i in range(num_chunks):
        this_args = [(arg[i] if is_chunked else arg) for (arg, is_chunked) in new_args]
        this_kwargs = {k: (arg[i] if is_chunked else arg) for (k, arg, is_chunked) in new_kwargs}
        output = func(*this_args, **this_kwargs)
        output_list.append(output)
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

def random_tiling(tiles: torch.Tensor, colors: int, kind: str, indices=(8, 8)):
    assert isinstance(kind, str)
    if isinstance(indices, torch.Tensor):
        has_batch = len(indices.shape) == 3
        pass
    elif isinstance(indices, np.ndarray):
        has_batch = len(indices.shape) == 3
        indices = torch.tensor(indices)
    else:
        if len(indices) == 2:
            has_batch = False
            h, w = indices
            b = 1
        elif len(indices) == 3:
            has_batch = True
            b, h, w = indices
        else:
            raise ValueError(f'can not interpret indicies with shape {indices.shape}')
        if kind == 'self':
            indices = random_wang_indicies(b, h, w, tiles.shape[0], kind)
        else:
            indices = random_wang_indicies(b, h, w, colors, kind)
    assert isinstance(indices, torch.Tensor)
    layers = index_simple_patches(indices, tiles, kind)
    blended = blend_patch_layers(indices, layers, kind)
    if not has_batch:
        blended = blended[0]
    return blended

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
        packed_edge_indicies = convolve_wang_indicies(packed_colors, colors, 'wang')
        packed_tile_counts = torch.zeros(colors**4, dtype=torch.int64)
        packed_tile_counts[packed_edge_indicies] += 1
        assert torch.all(packed_tile_counts == 1)

        return packed_edge_indicies
    elif kind == 'dual':
        packed_dual_indicies = convolve_wang_indicies(packed_colors, colors, 'dual')
        packed_tile_counts = torch.zeros(colors**4*2, dtype=torch.int64)
        packed_tile_counts[packed_dual_indicies] += 1
        assert torch.all(packed_tile_counts == 1)

        return packed_dual_indicies
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
        assert torch.allclose(random_tiling(unpermuted, 3, kind, perm), packed)
    
    return unpermuted

def pack_tiles(unpacked: torch.Tensor, colors: int, kind: str):
    if kind == 'wang' or kind == 'dual':
        perm = get_packing(colors, kind)
    elif kind == 'self':
        c = int(np.ceil(np.sqrt(unpacked.shape[0])))
        perm = torch.arange(c**2).reshape(1, 1, c, c) % unpacked.shape[0]
    else:
        raise NotImplementedError()
        
    return random_tiling(unpacked, 3, kind, perm)

textile_model = None

@torch.no_grad()
def select_candidates(tiles: torch.Tensor, source: torch.Tensor, candidates: int, keep: int, metric: str):
    tiles = rearrange(tiles, '(c t) k h w -> t c k h w', c=candidates)

    assert keep <= candidates

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
    tw = resolution*2
    th = resolution*2
    i, j = torch.meshgrid(
        torch.linspace(-diameter, diameter, steps=tw, dtype=torch.float32),
        torch.linspace(-diameter, diameter, steps=th, dtype=torch.float32),
        indexing='ij',
    )
    mask = ((torch.abs(i) + torch.abs(j)) < 1).to(source.dtype)
    return mask.reshape(1, tw, th)


def inpainting_from_boundaries(source: torch.Tensor, colors: int, boundary_kind: str, source_kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
            field = convolve_wang_indicies(field.unsqueeze(0), colors, 'wang')
            combined = index_simple_patches(field, source, 'wang')[0]
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
    elif (boundary_kind == 'dual' or boundary_kind == 'wang') and source_kind == 'dual':
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
            # (in fact, only the interior tiles probably exist at this point)
            field = convolve_wang_indicies(field.unsqueeze(0), colors, 'wang')[..., :2, :2]
            combined = index_simple_patches(field, source, 'wang')[0]
            tile_outsides.append(combined)
    
        tile_outsides = torch.cat(tile_outsides)
        outsides_mask = diamond_inpaint_mask(tile_size, source)
    else:
        raise NotImplementedError(f'can not make {boundary_kind} boundaries from {source_kind}')

    return (tile_outsides, outsides_mask)
