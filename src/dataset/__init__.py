from .single_asset_dataset import SingleAssetDataset

# Lazy import MultiAssetDataset to avoid 'datasets' library dependency
def __getattr__(name):
    if name == "MultiAssetDataset":
        from .multi_asset_dataset import MultiAssetDataset
        return MultiAssetDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SingleAssetDataset",
    "MultiAssetDataset"
]