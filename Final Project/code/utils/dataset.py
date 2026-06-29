"""Hugging Face dataset loading and PyTorch DataLoader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

from utils.transforms import RestorationTransforms


DAMAGED_COLUMN = "damaged_image"
TARGET_COLUMN = "pristine_image"


class OpenPhotoRestorationDataset:
    """PyTorch wrapper for paired damaged/pristine samples."""

    def __init__(self, dataset: Dataset, transform: RestorationTransforms) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        damaged = self.transform(sample[DAMAGED_COLUMN])
        pristine = self.transform(sample[TARGET_COLUMN])
        return {
            "damaged": damaged,
            "target": pristine,
            "index": index,
        }


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "joshuachin/openphoto-restore-dataset"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2
    validation_ratio: float = 0.1
    seed: int = 42
    train_subset: int | None = None
    val_subset: int | None = None
    test_subset: int | None = None


def load_openphoto_dataset(dataset_name: str = "joshuachin/openphoto-restore-dataset") -> DatasetDict:
    data = load_dataset(dataset_name)
    required_splits = {"train", "test"}
    missing_splits = required_splits.difference(data.keys())
    if missing_splits:
        raise ValueError(f"Dataset is missing required split(s): {sorted(missing_splits)}")

    required_columns = {DAMAGED_COLUMN, TARGET_COLUMN}
    for split_name in required_splits:
        missing_columns = required_columns.difference(data[split_name].column_names)
        if missing_columns:
            raise ValueError(f"{split_name} split is missing column(s): {sorted(missing_columns)}")
    return data


def split_train_validation(
    dataset: Dataset,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    split = dataset.train_test_split(test_size=validation_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]


def _maybe_select(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def create_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = load_openphoto_dataset(config.dataset_name)
    train_split, val_split = split_train_validation(data["train"], config.validation_ratio, config.seed)
    test_split = data["test"]

    train_split = _maybe_select(train_split, config.train_subset)
    val_split = _maybe_select(val_split, config.val_subset)
    test_split = _maybe_select(test_split, config.test_subset)

    transform = RestorationTransforms(image_size=config.image_size)
    train_dataset = OpenPhotoRestorationDataset(train_split, transform)
    val_dataset = OpenPhotoRestorationDataset(val_split, transform)
    test_dataset = OpenPhotoRestorationDataset(test_split, transform)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
