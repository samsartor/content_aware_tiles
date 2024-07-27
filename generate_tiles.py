#!/usr/bin/env python

from pathlib import Path
from tap import Tap

def str_or_int(x):
    try:
        return int(x)
    except ValueError:
        return x

class GenerateTilesArgs(Tap):
    dir: Path  # the input/output directory
    image: Path | None = None  # the source image
    prompt: str  # text prompt which describes the tiles to generate
    image_prompt: str | None = None # text prompt for the source image (if not provided), defaults to the same as --prompt
    neg_prompt: str = 'visible seams, indistinct, text, watermark' # negative text prompt to use with classifier-free guidance
    neg_image_prompt: str | None = None # negative text prompt for the source image (if not provided), defaults to the same as --neg_prompt
    inpaint_model: str = 'stabilityai/stable-diffusion-2-inpainting' # the model to use for tile inpainting
    image_model: str = 'stabilityai/stable-diffusion-xl-base-1.0' # the model to use to generate the source image (if not provided)
    kinds: list[str] = ['wang', 'dual', 'self', 'classic_wang', 'rolled_self'] # the kinds of tiles to generate
    colors: int = 3 # the number of edge colors for wang tilings or corner colors for dual tilings
    size: int = 256 # the resolution of each tile (must be 256 for SD2 inpainting or 512 for SDXL inpainting)
    cut_length: int | str = 256 # the length of each cut taken from the source image (can be any resolution less than the image size or "fill")
    cut_mode: str = 'ortho constrained' # the mode for choosing random cuts, either "ortho constrained" or "arbitrary"
    diffusion_steps: int = 40 # the number of diffusion steps
    diffusion_cfg: float = 7.5 # the classifier free guidance scale
    diffusion_sampler: str = 'euler_a' # the diffusion sampler to use
    diffusion_batch_size: int = 8 # the maximum batch size for the diffusion UNET and VAE (limits memory usage)
    candidates: int = 1 # the number of candidates to generate and choose from for each tile
    rejection_metric: str = 'sifid' # the metric to use for choosing tiles, either "sifid" or "textile"
    generate_tilings: bool = False # whether to save random tilings along with each packed tile set
    tiling_width: int = 28 # the number of tiles horizontally in each random tiling
    tiling_height: int = 10 # the number of tiles vertically in each random tiling
    rolled_image: bool = True # whether to use noise rolling when generating the source image (if not provided)
    rolled_seam_margin: float # the fraction of each tile that is inpainted when generating "rolledself" tiles
    self_tiles: int = 16 # the number of self-tiling tiles to generate for use in a stochastic tiling
    classic_overlap: int = 10 # the overlap between patches when running the classic graph-cut wang tile algorithm
    classic_attempts: int = 8 # the number of attempts to make running the classic graph-cut wang tile algorithm
    output_html: bool = True # whether to save an index.html file in the output directory
    device: str = 'cuda' # the pytorch device to use
    prefix: str | None = None # the prefix for every file saved in the output directory (defaults to the name of the directory)
    resume: bool = True  # resume generation using files previously saved in directory

    def configure(self):
        self.add_argument('dir', type=Path, help='the input/output directory')
        self.add_argument('--image', type=Path, help='the input image')
        self.add_argument('--prompt', type=str, required=True)
        self.add_argument('--kinds', type=lambda kinds: [k.strip() for k in kinds.split(',')])
        self.add_argument('--cut_length', type=str_or_int)
        self.add_argument('--rolled_seam_margin', type=eval, default='1/16')

opts = GenerateTilesArgs(explicit_bool=True).parse_args()

from content_aware_tiles.generation import inpainting_from_boundaries, place_tiles, random_cuts, pack_tiles, select_candidates, call_chunked, unpack_tiles, slide_wrapping, unslide_wrapping, diamond_inpaint_mask

import copy
import math
from functools import partial, lru_cache
import torch
from torch import Tensor
from einops import rearrange
from torchvision import io
from torchvision.transforms.functional import resize
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting, AutoPipelineForText2Image

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

samplers: dict[str, type[SchedulerMixin]] = {
    'euler': EulerDiscreteScheduler,
    'euler_a': EulerAncestralDiscreteScheduler,
    'dpm2': KDPM2DiscreteScheduler,
    'heun': HeunDiscreteScheduler,
    'ddim': DDIMScheduler,
}
if opts.diffusion_sampler in samplers:
    sampler = samplers[opts.diffusion_sampler]
else:
    print(f'{opts.diffusion_sampler} not one of {", ".join(samplers.keys())}')

def imload(path: Path) -> torch.Tensor:
    return (io.read_image(str(path)) / 255).to(torch.float16)

def imsave(img: torch.Tensor, path: Path):
    io.write_jpeg((img * 255).to(torch.uint8), str(path))

opts.dir.mkdir(exist_ok=True)
if opts.prefix is not None:
    prefix = opts.prefix
else:
    prefix = opts.dir.name + '.'

def rolling_pipe(pipe):
    # Create a copy of the pipe and its unet to avoid modifying the original
    pipe = copy.copy(pipe)
    pipe.unet = copy.copy(pipe.unet)

    # Store the original forward method
    old_forward = pipe.unet.forward

    # Define a new forward method that implements noise rolling
    def new_forward(x, *args, **kwargs):
        assert isinstance(x, torch.Tensor)
        _, _, w, h = x.shape
        
        # Generate random offsets for width and height
        off_w = torch.randint(0, w, [])
        off_h = torch.randint(0, h, [])

        # Apply sliding wrap to the input tensor
        x = slide_wrapping(x, off_w, off_h)

        # Process positional arguments
        new_args = []
        for v in args:
            # Apply sliding wrap to compatible tensor arguments
            if isinstance(v, torch.Tensor) and len(v.shape) == 4 and v.shape[-2] == w and v.shape[-1] == h:
                new_args.append(slide_wrapping(v, off_w, off_h))
            else:
                new_args.append(v)

        # Process keyword arguments
        new_kwargs = {}
        for k, v in kwargs.items():
            # Apply sliding wrap to compatible tensor keyword arguments
            if isinstance(v, torch.Tensor) and len(v.shape) == 4 and v.shape[-2] == w and v.shape[-1] == h:
                new_kwargs[k] = slide_wrapping(v, off_w, off_h)
            else:
                new_kwargs[k] = v

        # Call the original forward method with wrapped inputs
        x = old_forward(x, *new_args, **new_kwargs)[0]
        
        # Unwrap the output
        x = unslide_wrapping(x, off_w, off_h)
        return (x,)
    
    # Replace the unet's forward method with the new one
    pipe.unet.forward = new_forward
    return pipe

@lru_cache
def make_full_image():
    """
    Generate or load a full-size image exemplar for tile generation.
    """

    path = opts.dir / f'{prefix}synth_input.jpg'

    if opts.image is not None:
        print(f'📷 Loading {opts.image}')
        full_image = imload(opts.image)[None, ...]
    elif opts.resume and path.exists():
        print(f'💾 Loading {path}')
        full_image = imload(path)[None, ...]
    else:
        print(f'🫙 Loading {opts.image_model}')
        pipe = AutoPipelineForText2Image.from_pretrained(opts.image_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipe.to(opts.device)
        pipe.scheduler = sampler.from_config(pipe.scheduler.config) # type: ignore
        if opts.rolled_image:
            pipe = rolling_pipe(pipe)
        print(f'📷 Generating input image')
        full_image = pipe(
            prompt=opts.image_prompt or opts.prompt,
            negative_prompt=opts.neg_image_prompt or opts.neg_prompt,
            num_inference_steps=opts.diffusion_steps,
            guidance_scale=opts.diffusion_cfg,
            output_type="pt",
        ).images.cpu()
        if opts.rolled_image:
            full_image = torch.cat([full_image, full_image], -2)
            full_image = torch.cat([full_image, full_image], -1)
        imsave(full_image[0], path)

    return full_image

@lru_cache
def make_scaled_image():
    """
    Scales the generated exemplar image so that the boundary cuts are the correct length.
    """

    path = opts.dir / f'{prefix}scaled_input.jpg'

    if opts.resume and path.exists():
        print(f'💾 Loading {path}')
        scaled_image = imload(path)[None, ...]
    else:
        img = make_full_image()
        cut_length = opts.cut_length
        if cut_length == 'fill':
            cut_length = min(img.shape[-1], img.shape[-2])
        elif isinstance(cut_length, str):
            raise NotImplemented(f'cut_length={cut_length}')
        scaled_image = resize(
            img,
            [
                int(img.shape[-2] * opts.size / cut_length),
                int(img.shape[-1] * opts.size / cut_length),
            ],
        )
        imsave(scaled_image[0], path)

    return scaled_image

@lru_cache
def make_classic_wang_tiles():
    path = opts.dir / f'{prefix}classic_wang_tiles.jpg'

    if opts.resume and path.exists():
        print(f'💾 Loading {path}')
        classic_tiles = imload(path)
    else:
        scaled_image = (make_scaled_image() * 255).to(torch.uint8).numpy()[0]

        print('✂️ Creating classic tiles')
        from content_aware_tiles.classicwang import try_tilings

        classic_scaled_image = rearrange(scaled_image, 'c w h -> w h c')
        classic_tiles, _ = try_tilings(opts.colors, classic_scaled_image, opts.size, opts.classic_overlap, opts.classic_attempts)
        classic_tiles = rearrange(torch.tensor(classic_tiles) / 255, 'w h c -> c w h')
        imsave(classic_tiles, path)

    return unpack_tiles(classic_tiles, opts.colors, 'wang', opts.size)

@lru_cache
def make_inpainting_pipe():
    print(f'🫙 Loading {opts.inpaint_model}')
    pipe = AutoPipelineForInpainting.from_pretrained(opts.inpaint_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.to(opts.device)

    # Because our batch sizes are so large (eg 81 for 3 colors), we sub-batch the unet and vae 
    pipe.scheduler = sampler.from_config(pipe.scheduler.config) # type: ignore
    pipe.unet.forward = partial(call_chunked, pipe.unet.forward, chunk_size=opts.diffusion_batch_size)
    pipe.vae.decode = partial(call_chunked, pipe.vae.decode, chunk_size=opts.diffusion_batch_size)
    pipe.vae.encode = partial(call_chunked, pipe.vae.encode, chunk_size=opts.diffusion_batch_size)

    return pipe

def generate_tiles(pipe, scaled_image: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor, candidates: int, keep: int = 1):
    condition = condition.tile((candidates, 1, 1, 1))
    mask = mask[None, :, :, :].tile((condition.shape[0], 1, 1, 1))
    uncropped_tiles = pipe(
            prompt=opts.prompt,
            negative_prompt=opts.neg_prompt,
            num_inference_steps=opts.diffusion_steps,
            guidance_scale=opts.diffusion_cfg,
            image=condition,
            mask_image=mask,
            num_images_per_prompt=condition.shape[0],
            output_type="pt",
        ).images
    uncropped_tiles = select_candidates(uncropped_tiles, scaled_image.to(opts.device), candidates, keep, opts.rejection_metric).cpu()
    tiles = uncropped_tiles[:, :, opts.size//2:, opts.size//2:][:, :, :opts.size, :opts.size]
    return uncropped_tiles, tiles

@lru_cache
def make_self_tiles():
    path = opts.dir / f'{prefix}self_tiles.jpg'

    if opts.resume and path.exists():
        print(f'💾 Loading {path}')
        packed_self_tiles = imload(path)
        uncropped_self_tiles = None
        self_tiles = unpack_tiles(packed_self_tiles, opts.colors, 'self', opts.size, assert_roundtrip=True)
    else:
        pipe = make_inpainting_pipe()
        scaled_image = make_scaled_image()

        print('🖌️ Creating self-tiling tiles')
        cuts = random_cuts(scaled_image, opts.size, 1, opts.cut_mode)
        condition, mask = inpainting_from_boundaries(cuts, 1, 'self', 'cuts')
        uncropped_self_tiles, self_tiles = generate_tiles(pipe, scaled_image, condition, mask, opts.self_tiles*opts.candidates, opts.self_tiles)
        packed_self_tiles = pack_tiles(self_tiles, opts.colors, 'self')
        
        imsave(packed_self_tiles, path)

    return uncropped_self_tiles, self_tiles

@lru_cache
def make_wang_tiles():
    path = opts.dir / f'{prefix}wang_tiles.jpg'

    if opts.resume and path.exists():
        print(f'💾 Loading {path}')
        packed_wang_tiles = imload(path)
        uncropped_wang_tiles = None
        wang_tiles = unpack_tiles(packed_wang_tiles, opts.colors, 'wang', opts.size, assert_roundtrip=True)
    else:
        pipe = make_inpainting_pipe()
        scaled_image = make_scaled_image()
        print('🖌️ Creating wang tiles')

        cuts = random_cuts(scaled_image, opts.size, opts.colors, opts.cut_mode)
        condition, mask = inpainting_from_boundaries(cuts, opts.colors, 'wang', 'cuts')
        uncropped_wang_tiles, wang_tiles = generate_tiles(pipe, scaled_image, condition, mask, opts.candidates)
        packed_wang_tiles = pack_tiles(wang_tiles, opts.colors, 'wang')
        
        imsave(packed_wang_tiles, path)

    return uncropped_wang_tiles, wang_tiles

@lru_cache
def make_dual_tiles():
    path = opts.dir / f'{prefix}dual_tiles.jpg'

    if opts.resume and path.exists():
        print(f'💾 Loading {path}')
        packed_dual_tiles = imload(path)
        dual_tiles = unpack_tiles(packed_dual_tiles, opts.colors, 'dual', opts.size, assert_roundtrip=True)
    else:
        pipe = make_inpainting_pipe()
        scaled_image = make_scaled_image()
        uncropped_wang_tiles, wang_tiles = make_wang_tiles()

        print('🖌️ Creating dual (interior) tiles')
        if uncropped_wang_tiles is not None:
            condition, mask = inpainting_from_boundaries(uncropped_wang_tiles, opts.colors, 'dual', 'uncropped_wang')
        else:
            assert wang_tiles is not None
            condition, mask = inpainting_from_boundaries(wang_tiles, opts.colors, 'dual', 'wang')
        uncropped_interior_tiles, interior_tiles = generate_tiles(pipe, scaled_image, condition, mask, opts.candidates)
        print('🖌️ Creating dual (cross) tiles')
        condition, mask = inpainting_from_boundaries(interior_tiles, opts.colors, 'dual', 'dual')
        uncropped_cross_tiles, cross_tiles = generate_tiles(pipe, scaled_image, condition, mask, opts.candidates)
        dual_tiles = torch.cat([interior_tiles, cross_tiles], 0)

        packed_dual_tiles = pack_tiles(dual_tiles, opts.colors, 'dual')    
        imsave(packed_dual_tiles, path)

    return dual_tiles

@lru_cache
def make_rolled_self():
    path = opts.dir / f'{prefix}rolled_self_tiles.jpg'

    if opts.resume and path.exists():
        print(f'💾 Loading {path}')
        packed_self_tiles = imload(path)
        self_tiles = unpack_tiles(packed_self_tiles, opts.colors, 'self', opts.size, assert_roundtrip=True)
    else:
        pipe = rolling_pipe(make_inpainting_pipe())
        scaled_image = make_scaled_image()

        print('🛞 Creating noise-rolled self-tiling tile')

        _, _, w, h = scaled_image.shape
        off_w = 0 if w == opts.size else torch.randint(0, w-opts.size, [])
        off_h = 0 if h == opts.size else torch.randint(0, h-opts.size, [])
        condition = scaled_image[:, :, off_w:, off_h:][:, :, :opts.size, :opts.size]
        condition = resize(condition, [opts.size*2, opts.size*2]) # because we don't have boundaries, double the size
        condition = condition.tile((opts.candidates, 1, 1, 1))
        mask = torch.ones_like(condition)[:, :1, :, :]
        b = int(w * opts.rolled_seam_margin)
        mask[:, :, b:-b, b:-b] = 0

        self_tiles = pipe(
            prompt=opts.prompt,
            negative_prompt=opts.neg_prompt,
            num_inference_steps=opts.diffusion_steps,
            guidance_scale=opts.diffusion_cfg,
            image=condition,
            mask_image=mask,
            num_images_per_prompt=condition.shape[0],
            output_type="pt",
        ).images
        self_tiles = resize(self_tiles, [opts.size, opts.size])
        self_tiles = select_candidates(self_tiles, scaled_image.to(opts.device), opts.candidates, 1, opts.rejection_metric).cpu()

        imsave(self_tiles[0, 0], path)

    return self_tiles


def make_tiling(tiles: torch.Tensor, kind: str, name: str):
    if opts.generate_tilings:
        tiling = place_tiles(tiles.cpu(), opts.colors, kind, indices=(opts.tiling_height, opts.tiling_width))
        imsave(tiling, opts.dir / f'{prefix}{name}_tiling.jpg')

def make_page():
    cwd = opts.dir
    name = cwd.name
    prefix = opts.prefix or f"{name}."
    
    sqrt_self_tiles = int(math.ceil(math.sqrt(opts.self_tiles)))

    input_title = 'Input Image' if opts.image else 'Generated Image'
    cuts_title = 'Constrained Random Cuts' if opts.cut_mode == 'ortho constrained' else 'Random Cuts'
    style = """
    <style>
        body { margin: 50px; display: flex; flex-direction: column; gap: 15px; }
        .box { display: flex; flex-direction: column; align-items: center; border: 2px solid black; padding: 15px; }
        .prompt { text-align: center; font-style: italic; font-size: 32px; margin: 0 100px; }
        h2 { margin: 0; text-align: center; }
        .grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        tiling-canvas { width: 100%; height: 640px; }
        img { max-width: 100%; height: auto; }
    </style>
    """

    with (cwd / 'index.html').open('w') as o:
        print(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{opts.prompt}</title>
            <script type="module">{(Path(__file__).parent / 'tiling_canvas/dist/tiling_canvas.js').read_text()}</script>
            {style}
        </head>
        <body>
            <span class="prompt"><q>{opts.prompt}</q></span>
        """, file=o)

        # Generated tilings
        for kind in opts.kinds:
            tiles_image = f"{prefix}{kind}_tiles.jpg"
            mode = kind.split('_')[-1]
            if kind == 'rolled_self':
                mode = 'single'
            if mode == 'self':
                attrs = f'selftiles="{sqrt_self_tiles},{sqrt_self_tiles}"'
            else:
                attrs = ''
            if (cwd / tiles_image).exists():
                print(f"""
                <div class="box">
                    <h2>{kind.replace('_', ' ').title()} Tiling</h2>
                    <tiling-canvas src="{tiles_image}" mode="{mode}" alt="{kind.replace('_', ' ').title()} Tiling" {attrs}/>
                </div>
                """, file=o)
                
        # Input image        
        print('<div class="grid">', file=o)
        input_image = f"{prefix}scaled_input.jpg" if (cwd / f"{prefix}scaled_input.jpg").exists() else f"{prefix}synth_input.jpg"
        if (cwd / input_image).exists():
            print(f"""
            <div class="box">
                <h2>{input_title}</h2>
                <img src="{input_image}" alt="Input Image">
                <p>({cuts_title})</p>
            </div>
            """, file=o)

        # Packed tiles
        for kind in opts.kinds:
            tiles_image = f"{prefix}{kind}_tiles.jpg"
            if (cwd / tiles_image).exists():
                print(f"""
                <div class="box">
                    <h2>{kind.replace('_', ' ').title()} Tiles</h2>
                    <img src="{tiles_image}" alt="{kind.replace('_', ' ').title()} Tiles">
                </div>
                """, file=o)
        print('</div>', file=o)
        print('</body></html>', file=o)

if 'classic_wang' in opts.kinds:
    make_tiling(make_classic_wang_tiles(), 'wang', 'classic_wang')

if 'rolled_self' in opts.kinds:
    make_tiling(make_rolled_self(), 'self', 'rolled_self')

if 'self' in opts.kinds:
    make_tiling(make_self_tiles()[1], 'self', 'self')

if 'wang' in opts.kinds:
    make_tiling(make_wang_tiles()[1], 'wang', 'wang')

if 'dual' in opts.kinds:
    make_tiling(make_dual_tiles(), 'dual', 'dual')

if opts.output_html:
    make_page()
