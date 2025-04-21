# Content-aware Tile Generation using Exterior Boundary Inpaintings <br> [[Paper](https://bin.samsartor.com/content_aware_tiles.pdf)] [[Website](https://samsartor.com/content-aware-tiles)] [[Supplemental](https://bin.samsartor.com/content_aware_tile_supplemental/index.html)]

![visualization of tiling](doc/header_image.webp)

We present a novel and flexible learning-based method for generating
tileable image sets.  Our method goes beyond simple self-tiling,
supporting sets of mutually tileable images that exhibit a high
degree of diversity.  To promote diversity we decouple structure
from content by foregoing explicit copying of patches from an
exemplar image.  Instead we leverage the prior knowledge of natural
images and textures embedded in large-scale pretrained diffusion
models to guide tile generation constrained by exterior boundary
conditions and a text prompt to specify the content. By carefully
designing and selecting the exterior boundary conditions, we can
reformulate the tile generation process as an inpainting problem,
allowing us to directly employ existing diffusion-based inpainting
models without the need to retrain a model on a custom training set.
We demonstrate the flexibility and efficacy of our content-aware
tile generation method on different tiling schemes, such as Wang
tiles, from only a text prompt.  Furthermore, we introduce a novel
Dual Wang tiling scheme that provides greater texture continuity and
diversity than existing Wang tile variants.

```bash
$ uv run generate_tiles out/orange_lily \
    --image doc/orange_lily.input.jpg \
    --prompt "bright orange lily in a flower garden, small blue and white flowers, leaves"

$ open out/orange_lily/orange_lily.dual_tiles.jpg
```
<img alt="orange lilies example output" src="doc/orange_lily.dual_tiles.jpg" width="300px">

# Installation
## ComfyUI

1. checkout the repository in your "custom_nodes" directory with `git clone https://github.com/samsartor/content_aware_tiles`
2. (optional) `pip install -e requirements.txt`` in your "content_aware_tiles" directory to install extra dependencies
3. Load the "content_aware_tiles_workflow.json" in ComfyUI

![ComfyUI screenshot](doc/workflow_screenshot.png)

## Command Line (with uv)

1. checkout the repository anywhere with `git clone https://github.com/samsartor/content_aware_tiles` and `cd content_aware_tiles`
2. run `uv run generate_tiles.py [OUTPUT_DIRECTORY] --prompt "[PROMPT]"`

On an Apple you should also pass `--device mps`

## Command Line (with venv)

1. checkout the repository anywhere with `git clone https://github.com/samsartor/content_aware_tiles` and `cd content_aware_tiles`
2. create a virtual environment with `python -m venv venv && source venv/bin/activate`
3. install with `pip install -e '.'`
4. run `generate_wang [OUTPUT_DIRECTORY] --prompt "[PROMPT]"`

## Docker

1. `docker build . -t content_aware_tiles`
2. `docker compose -f compose.yaml run --rm content_aware_tiles`
3. (optional) `huggingface-cli download stabilityai/stable-diffusion-2-inpainting 512-inpainting-ema.safetensors`
4. either `python ComfyUI/main.py` or `generate_tiles`

The docker image only supports cuda.

# Usage
## Positional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dir` | *Required* | The input/output directory |

## Basic Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | *Required* | Text prompt which describes the tiles to generate |
| `--image` | `None` | The source image |

## Model Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--inpaint_model` | `"stabilityai/stable-diffusion-2-inpainting"` | The model to use for tile inpainting |
| `--image_model` | `"stabilityai/stable-diffusion-xl-base-1.0"` | The model to use to generate the source image (if `--image` is not provided) |
| `--device` | `"cuda"` | The pytorch device to use |

## Tile Generation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--kinds` | `"wang,dual,self,classic_wang,rolled_self"` | The kinds of tiles to generate |
| `--colors` | `3` | The number of edge colors for wang tilings or corner colors for dual tilings |
| `--size` | `256` | The resolution of each tile (must be 256 for SD2 inpainting or 512 for SDXL inpainting) |
| `--cut_length` | `1/2` | The size of each tile as a fraction of the source image |
| `--cut_mode` | `"ortho constrained"` | The mode for choosing random cuts, either "ortho constrained" or "arbitrary" |

## Diffusion Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--diffusion_steps` | `40` | The number of diffusion steps |
| `--diffusion_cfg` | `7.5` | The classifier free guidance scale |
| `--diffusion_sampler` | `"euler_a"` | The diffusion sampler to use |
| `--diffusion_batch_size` | `8` | The maximum batch size for the diffusion UNET and VAE (limits memory usage) |

## Selection and Tiling Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--candidates` | `1` | The number of candidates to generate and choose from for each tile |
| `--rejection_metric` | `"sifid"` | The metric to use for choosing tiles, either "sifid" or "textile" |
| `--generate_tilings` | `False` | Whether to save random tilings along with each packed tile set |
| `--tiling_width` | `28` | The number of tiles horizontally in each random tiling |
| `--tiling_height` | `10` | The number of tiles vertically in each random tiling |

## Advanced Tile Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--rolled_image` | `True` | Whether to use noise rolling when generating the source image (if `--image` is not provided) |
| `--rolled_seam_margin` | `1/16` | The fraction of each tile that is inpainted when generating "rolledself" tiles |
| `--self_tiles` | `16` | The number of self-tiling tiles to generate for use in a stochastic tiling |
| `--classic_overlap` | `10` | The overlap between patches when running the classic graph-cut wang tile algorithm |
| `--classic_attempts` | `8` | The number of attempts to make running the classic graph-cut wang tile algorithm |
| `--image_prompt` | Same as `--prompt` | Text prompt for the source image (if `--image` is not provided) |
| `--neg_prompt` | `"visible seams, indistinct, text, watermark"` | Negative text prompt to use with classifier-free guidance |
| `--neg_image_prompt` | Same as `--neg_prompt` | Negative text prompt for the source image (if `--image` is not provided) |

## Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_html` | `True` | Whether to save an index.html file in the output directory |
| `--prefix` | `None` | The prefix for every file saved in the output directory (defaults to the name of the directory) |
| `--resume` | `True` | Resume generation using files previously saved in directory |
| `-h, --help` | - | Show this help message and exit |

# Citation

```bibtex
@conference{Sartor:2024:CAT,
    author    = {Sartor, Sam and Peers, Pieter},
    title     = {Content-aware Tile Generation using Exterior Boundary Inpainting},
    month     = {December},
    year      = {2024},
    booktitle = {ACM Transactions on Graphics},
}
```
