from dataclasses import dataclass, replace
import torch
import numpy as np
import nodes
from functools import partial
import copy
from einops import rearrange
from .generation import unpack_tiles, pack_tiles, slide_wrapping, unslide_wrapping, call_chunked, inpainting_from_boundaries, random_cuts, select_candidates, place_tiles

@dataclass
class Tileset:
    kind: str
    colors: int
    resolution: int
    candidates: int

    def latent(self, vae) -> 'Tileset':
        return replace(self, resolution=self.resolution // vae.downscale_ratio)

class RandomWangTileBoundaries:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["ortho constrained", "arbitrary"],),
                "colors": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "candidates": ("INT", {"default": 1}),
                "resolution": ("INT", {"default": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TILESET")
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, image: torch.Tensor, mode: str, colors: int, candidates: int, resolution: int):
        source = image.permute(0, 3, 1, 2)
        cuts = random_cuts(source, resolution, colors, mode)
        tileset = Tileset(
            kind='wang' if colors > 1 else 'self',
            colors=colors,
            resolution=resolution,
            candidates=candidates,
        )
        tile_outsides, outsides_mask = inpainting_from_boundaries(cuts, tileset.colors, tileset.kind, 'cuts')
        tile_outsides = tile_outsides.tile((candidates, 1, 1, 1)).permute(0, 2, 3, 1)
        outsides_mask = outsides_mask.tile((tile_outsides.shape[0], 1, 1))
        return [
            tile_outsides,
            outsides_mask,
            tileset,
        ]

def tilize_tiles(image: torch.Tensor, tileset: Tileset):
    b, c, w, h = image.shape
    res = tileset.resolution
    if tileset.kind == 'self' and b == 1:
        return unpack_tiles(image[0, ...], tileset.colors, tileset.kind, res)    
    elif b == 1 and w == tileset.colors**2*res and h == tileset.colors**2*res:
        return unpack_tiles(image[0, ...], tileset.colors, tileset.kind, res)
    else:
        if tileset.kind != 'self':
            assert b in [tileset.candidates*tileset.colors**4, tileset.candidates*tileset.colors**4*2], f'tileset {tileset} can not have {b} tiles'
        if w == res and h == res:
            # cropped tiles
            return image
        elif w == res*2 and h == res*2:
            # uncropped tiles
            crop = res//2
            return image[:, :, crop:, crop:][:, :, :res, :res]
    raise ValueError(f'tileset {tileset} can not have image shape {image.shape}')

class WangBoundaries:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tileset": ("TILESET", {"forceInput": True}),
                "candidates": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TILESET")
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, tiles: torch.Tensor, tileset: Tileset, candidates: int):
        new_tileset = Tileset(
            kind='wang',
            colors=tileset.colors,
            resolution=tileset.resolution,
            candidates=candidates,
        )
        tiles = tiles.permute(0, 3, 1, 2)
        if tileset.kind == 'wang' and tiles.shape[-1] == tileset.resolution*2:
            # special case, use the uncropped tiles
            tile_outsides, outsides_mask = inpainting_from_boundaries(tiles, new_tileset.colors, new_tileset.kind, 'uncropped_wang')
        else:
            tiles = tilize_tiles(tiles, tileset)
            tile_outsides, outsides_mask = inpainting_from_boundaries(tiles, new_tileset.colors, new_tileset.kind, tileset.kind)
        tile_outsides = tile_outsides.permute(0, 2, 3, 1).tile((candidates, 1, 1, 1))
        outsides_mask = outsides_mask.tile((tile_outsides.shape[0], 1, 1))
        return [
            tile_outsides,
            outsides_mask,
            new_tileset,
        ]
        
class DualBoundaries:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tileset": ("TILESET", {"forceInput": True}),
                "candidates": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TILESET")
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, tiles: torch.Tensor, tileset: Tileset, candidates: int):
        new_tileset = Tileset(
            kind='dual',
            colors=tileset.colors,
            resolution=tileset.resolution,
            candidates=candidates,
        )
        tiles = tiles.permute(0, 3, 1, 2)
        if tileset.kind == 'wang' and tiles.shape[-1] == tileset.resolution*2:
            # special case, use the uncropped tiles
            tile_outsides, outsides_mask = inpainting_from_boundaries(tiles, new_tileset.colors, new_tileset.kind, 'uncropped_wang')
        else:
            tiles = tilize_tiles(tiles, tileset)
            tile_outsides, outsides_mask = inpainting_from_boundaries(tiles, new_tileset.colors, new_tileset.kind, tileset.kind)
        tile_outsides = tile_outsides.permute(0, 2, 3, 1).tile((candidates, 1, 1, 1))
        outsides_mask = outsides_mask.tile((tile_outsides.shape[0], 1, 1))
        return [
            tile_outsides,
            outsides_mask,
            new_tileset,
        ]

class LatentDualBoundaries:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("LATENT",),
                "tileset": ("TILESET", {"forceInput": True}),
                "vae": ("VAE",),
                "candidates": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("LATENT", "TILESET")
    FUNCTION = "generate"
    CATEGORY = "tiles/latent"

    def generate(self, tiles: dict[str, torch.Tensor], tileset: Tileset, vae, candidates: int):
        new_tileset = Tileset(
            kind='dual',
            colors=tileset.colors,
            resolution=tileset.resolution,
            candidates=candidates,
        )
        x = tiles['samples']
        if tileset.kind == 'wang' and x.shape[-1] == tileset.resolution*2:
            # special case, use the uncropped tiles
            tile_outsides, outsides_mask = inpainting_from_boundaries(x, new_tileset.colors, new_tileset.kind, 'uncropped_wang')
        else:
            x = tilize_tiles(x, tileset)
            tile_outsides, outsides_mask = inpainting_from_boundaries(x, new_tileset.colors, new_tileset.kind, tileset.kind)
        tile_outsides = tile_outsides.tile((candidates, 1, 1, 1))
        outsides_mask = outsides_mask[:, None, :, :].tile((x.shape[0], x.shape[1], 1, 1))
        return [
            {
                'samples': tile_outsides,
                'noise_mask': outsides_mask,
            },
            new_tileset,
        ]

class RejectCandidateTiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tileset": ("TILESET", {"forceInput": True}),
                "source": ("IMAGE",),
                "metric": (["sifid"],),
                "keep": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILESET")
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, tiles: torch.Tensor, tileset: Tileset, source: torch.Tensor, metric: str, keep: int):
        if tileset.candidates == keep:
            return [tiles, tileset]

        tiles = rearrange(tiles, 't h w k -> t k h w').cuda()
        source = rearrange(source, 'b h w k -> b k h w').cuda()

        selected_tiles = select_candidates(tiles, source, tileset.candidates, keep, metric)

        selected_tiles = rearrange(selected_tiles, 't k h w -> t h w k').cpu()
        return [
            selected_tiles,
            replace(tileset, candidates=keep),
        ]

    
class TilePacking:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tileset": ("TILESET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, tiles: torch.Tensor, tileset: Tileset):
        x = tiles.permute(0, 3, 1, 2)
        x = tilize_tiles(x, tileset)
        if tileset.kind == 'dual' and x.shape[0] != tileset.colors**4*2:
            raise ValueError(f'must generate interior and cross tiles ({tileset.colors**4*2} total), found {x.shape[0]} tiles')
        x = pack_tiles(x, tileset.colors, tileset.kind)
        x = x[None, :, :, :].permute(0, 2, 3, 1)
        return [x]

class TileUnpacking:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "packing": ("IMAGE",),
                "kind": (["wang", "dual"],),
                "resolution": ("INT", {"default": 256}),
                "colors": ("INT", {"default": 3}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILESET")
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, packing: torch.Tensor, kind: str, resolution: int, colors: int):
        tileset = Tileset(
            kind=kind,
            colors=colors,
            resolution=resolution,
            candidates=1,
        )
        packing = packing.permute(0, 3, 1, 2)
        tiles = tilize_tiles(packing, tileset)
        tiles = tiles.permute(0, 2, 3, 1)
        return [tiles, tileset]

class LatentTilePacking:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("LATENT",),
                "tileset": ("TILESET", {"forceInput": True}),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "tiles/latent"

    def generate(self, tiles: dict[str, torch.Tensor], tileset: Tileset, vae):
        x = tiles['samples']
        x = tilize_tiles(x, tileset.latent(vae))
        if tileset.kind == 'dual' and x.shape[0] != tileset.colors**4*2:
            raise ValueError(f'must generate interior and cross tiles ({tileset.colors**4*2} total), found {x.shape[0]} tiles')
        x = pack_tiles(x, tileset.colors, tileset.kind)
        x = x[None, :, :, :]
        return [{'samples': x}]


class RandomTiling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tileset": ("TILESET", {"forceInput": True}),
                "width": ("INT", {"min": 1, "default": 20}),
                "height": ("INT", {"min": 1, "default": 20}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "tiles"

    def generate(self, tiles: torch.Tensor, tileset: Tileset, width: int, height: int):
        x = tiles.permute(0, 3, 1, 2)
        x = tilize_tiles(x, tileset)
        if tileset.kind == 'dual' and x.shape[0] != tileset.colors**4*2:
            raise ValueError(f'must generate interior and cross tiles ({tileset.colors**4*2} total), found {x.shape[0]} tiles')
        x = place_tiles(x, tileset.colors, tileset.kind, indices=(1, height, width))
        x = x.permute(0, 2, 3, 1)
        return [x]


class LatentRandomTiling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("LATENT",),
                "tileset": ("TILESET", {"forceInput": True}),
                "vae": ("VAE",),
                "width": ("INT", {"min": 1, "default": 20}),
                "height": ("INT", {"min": 1, "default": 20}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "tiles/latent"

    def generate(self, tiles: dict[str, torch.Tensor], tileset: Tileset, vae, width: int, height: int):
        x = tiles['samples']
        x = tilize_tiles(x, tileset.latent(vae))
        if tileset.kind == 'dual' and x.shape[0] != tileset.colors**4*2:
            raise ValueError(f'must generate interior and cross tiles ({tileset.colors**4*2} total), found {x.shape[0]} tiles')
        x = place_tiles(x, tileset.colors, tileset.kind, indices=(1, height, width))
        return [x]


class TileImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tileset": ("TILESET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "tiles"
    
    def generate(self, tiles: torch.Tensor, tileset: Tileset):
        tiles = tiles.permute(0, 3, 1, 2)
        tiles = tilize_tiles(tiles, tileset)
        tiles = tiles.permute(0, 2, 3, 1)
        return [tiles]

class LatentTileImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("LATENT",),
                "tileset": ("TILESET", {"forceInput": True}),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "tiles/latent"
    
    def generate(self, tiles: dict[str, torch.Tensor], tileset: Tileset, vae):
        x = tilize_tiles(tiles['samples'], tileset.latent(vae))
        return [{'samples': x}]


class SubBatchModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "subbatch_size": ("INT", {"default": 8}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    def patch(self, model, subbatch_size: int):
        new_model = model.clone()
        # we should really use set_model_unet_function like in RollingKSampler, but then it won't compose
        new_model.add_object_patch('apply_model', partial(call_chunked, new_model.model.apply_model, chunk_size=subbatch_size))
        return [new_model]


class SubBatchControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CONTROL_NET",),
                "subbatch_size": ("INT", {"default": 8}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    def patch(self, model, subbatch_size: int):
        new_model = model.copy()
        new_model.get_control = partial(call_chunked, new_model.get_control, chunk_size=subbatch_size)
        return [new_model]


class SubBatchVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VAE",),
                "subbatch_size": ("INT", {"default": 8}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    def patch(self, model, subbatch_size: int):
        new_model = copy.copy(model)
        new_model.encode = partial(call_chunked, new_model.encode, chunk_size=subbatch_size)
        new_model.decode = partial(call_chunked, new_model.decode, chunk_size=subbatch_size)
        return [new_model]


class RollingKSampler(nodes.KSampler):
    CATEGORY = "tiles"

    @classmethod
    def INPUT_TYPES(s):
        tys = super(RollingKSampler, s).INPUT_TYPES()
        tys['required']['double_output'] = ('BOOLEAN', {'default': True})
        return tys

    def sample(self, model, seed, double_output=True, **kwargs):
        generator = torch.manual_seed(seed+1)
        
        def roll_inputs(func, kwargs: dict):
            x = kwargs['input']
            t = kwargs['timestep']
            c = kwargs['c']
            assert isinstance(x, torch.Tensor)
            assert isinstance(c, dict)
            _, _, w, h = x.shape
            off_w = torch.randint(0, w, [], generator=generator)
            off_h = torch.randint(0, h, [], generator=generator)

            x = slide_wrapping(x, off_w, off_h)

            for k, v in c.items():
                if isinstance(v, torch.Tensor) and len(v.shape) == 4 and v.shape[-2] == w and v.shape[-1] == h:
                    c[k] = slide_wrapping(v, off_w, off_h)

            x = func(x, t, **c)
            x = unslide_wrapping(x, off_w, off_h)
            return x
        
        model = model.clone()
        model.set_model_unet_function_wrapper(roll_inputs)
        x = super().sample(model, seed, **kwargs)[0]['samples']
        if double_output:
            x = torch.cat([x, x], -1)
            x = torch.cat([x, x], -2)
        return [{'samples': x}]
    
class RandomSubsetOfBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select"
    CATEGORY = "image/batch"
    
    def select(self, image: torch.Tensor, count: int):
        return [image[torch.randint(image.shape[0], [count]), :, :, :]]
