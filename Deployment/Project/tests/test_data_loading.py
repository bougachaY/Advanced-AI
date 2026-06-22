"""Disk-loading tests for the training data paths."""

from types import SimpleNamespace

from datasets import Dataset, DatasetDict, load_from_disk

import train
from data.dataset import CauldronDataset, FlickrDataset
from models.config import ProjectorConfig, TrainConfig, VLMConfig, ViTConfig


class FakeTokenizer:
    pad_token_id = 0
    image_token = "<|image|>"
    image_token_id = 1


def tiny_vlm_cfg():
    return VLMConfig(
        vit=ViTConfig(img_size=32),
        projector=ProjectorConfig(image_token_length=4),
        load_backbone_weights=False,
    )


def patch_processors(monkeypatch):
    monkeypatch.setattr(train, "get_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(train, "get_image_processor", lambda *args, **kwargs: object())


def test_train_loads_flickr_dataset_from_disk(tmp_path, monkeypatch):
    patch_processors(monkeypatch)
    disk_path = tmp_path / "flickr30k"
    DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "image": "image-placeholder",
                        "original_alt_text": ["a caption"],
                    }
                ]
            )
        }
    ).save_to_disk(disk_path)
    train_cfg = TrainConfig(
        dataset_type="flickr",
        dataset_local_path=str(disk_path),
        batch_size=1,
    )

    train_loader, val_loader = train.get_dataloaders(train_cfg, tiny_vlm_cfg())

    assert isinstance(train_loader.dataset, FlickrDataset)
    assert isinstance(val_loader.dataset, FlickrDataset)
    assert len(train_loader.dataset.dataset) == 1
    assert len(val_loader.dataset.dataset) == 1


def test_train_loads_cauldron_subsets_from_disk(tmp_path, monkeypatch):
    patch_processors(monkeypatch)
    base_path = tmp_path / "the_cauldron"
    subset_path = base_path / "ai2d"
    DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "images": ["image-placeholder"],
                        "texts": [{"user": "question", "assistant": "answer"}],
                    }
                ]
            )
        }
    ).save_to_disk(subset_path)
    train_cfg = TrainConfig(
        dataset_type="cauldron",
        dataset_local_path=str(base_path),
        dataset_subsets=("ai2d", "missing_subset"),
        batch_size=1,
    )

    train_loader, val_loader = train.get_dataloaders(train_cfg, tiny_vlm_cfg())

    assert isinstance(train_loader.dataset, CauldronDataset)
    assert isinstance(val_loader.dataset, CauldronDataset)
    assert len(train_loader.dataset.dataset) == 1
    assert len(val_loader.dataset.dataset) == 1


def test_train_mmstar_validation_loads_val_split_from_disk(tmp_path):
    disk_path = tmp_path / "mmstar"
    DatasetDict(
        {
            "val": Dataset.from_list(
                [
                    {
                        "question": "What is shown?",
                        "answer": "A",
                        "category": "coarse-perception",
                    }
                ]
            )
        }
    ).save_to_disk(disk_path)

    raw_mmstar = load_from_disk(str(disk_path))
    mmstar_val = raw_mmstar["val"] if "val" in raw_mmstar else raw_mmstar

    assert len(mmstar_val) == 1
    assert mmstar_val[0]["question"] == "What is shown?"
