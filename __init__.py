try:
    import comfy
    from .content_aware_tiles.comfyui_nodes import *

    NODE_CLASS_MAPPINGS = {
        "RandomWangBoundaries": RandomWangTileBoundaries,
        "WangBoundaries": WangBoundaries,
        "DualBoundaries": DualBoundaries,
        "RejectCandidateTiles": RejectCandidateTiles,
        "TileImages": TileImages,
        "TilePacking": TilePacking,
        "TileUnpacking": TileUnpacking,
        "LatentDualBoundaries": LatentDualBoundaries,
        "LatentTileImages": LatentTileImages,
        "LatentTilePacking": LatentTilePacking,
        "RandomTiling": RandomTiling,
        "LatentRandomTiling": LatentRandomTiling,
        "RollingKSampler": RollingKSampler,
        "SubBatchModel": SubBatchModel,
        "SubBatchVAE": SubBatchVAE,
        "RandomSubsetOfBatch": RandomSubsetOfBatch,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "RandomWangBoundaries": "Random Wang Boundaries (from Image)",
        "WangBoundaries": "Wang Boundaries (from Tiles)",
        "DualBoundaries": "Dual Boundaries",
        "RejectCandidateTiles": "Reject Candidate Tiles",
        "TileImages": "Tile Images",
        "TilePacking": "Tile Packing",
        "TileUnpacking": "Tile Unpacking",
        "LatentDualBoundaries": "Dual Boundaries (Latent)",
        "LatentTileImages": "Tile Images (Latent)",
        "LatentTilePacking": "Tile Packing (Latent)",
        "RandomTiling": "Random Tiling",
        "LatentRandomTiling": "Random Tiling (Latent)",
        "RollingKSampler": "KSampler (Rolling)",
        "SubBatchModel": "Sub-batching Model",
        "SubBatchVAE": "Sub-batching VAE",
        "RandomSubsetOfBatch": "Random Subset of Batch",
    }
except ImportError as e:
    if e.name != 'comfy':
        raise e
