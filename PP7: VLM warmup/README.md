# PP7: VLM warmup

This practical is a minimal warmup for the VLM project.

The goal is to:

- load a pretrained vision tower and a pretrained language tower directly from Hugging Face,
- implement the modality projector,
- implement the multimodal glue that replaces image placeholder tokens with projected vision features,
- train for a few minutes on a small subset of Flickr30k,
- run generation locally to verify that the code executes end-to-end.

Only the following files are meant to contain student work:

- `models/modality_projector.py`
- `models/vision_language_model.py`

Reference solutions are provided in:

- `models/modality_projector_solution.py`
- `models/vision_language_model_solution.py`

## Install

From the repository root:

```bash
uv sync --group student
```

## Dataset Cache

`train.py` now reads Flickr30k through the standard Hugging Face datasets cache.
If the dataset has already been downloaded on a shared filesystem, point students to that cache with `--dataset-cache-dir`.
If you want to pre-populate that shared cache yourself, you can use `download.py` with the same `--dataset-cache-dir`.
If you want the scripts to fail fast instead of attempting network access, export `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` in the job environment.

## Train

Example:

```bash
cd "PP7: VLM warmup"
uv run python train.py \
  --dataset-path AnyModal/flickr30k \
  --dataset-cache-dir /path/to/hf-datasets-cache \
  --train-samples 256 \
  --val-samples 64 \
  --eval-interval 10 \
  --max-steps 20
```

The default setup freezes SigLIP and SmolLM and only trains the modality projector.
`--dataset-cache-dir` is passed directly to `datasets.load_dataset`, so it should point at the shared cache location where Flickr30k was downloaded.
`train.py` now keeps a held-out validation split and reports both smoothed train loss and validation loss during training.
Once `models/modality_projector.py` and `models/vision_language_model.py` are implemented, `train.py` should run as-is.

## Generate

After training:

```bash
uv run python generate.py \
  --checkpoint checkpoints/projector.pt \
  --image /path/to/image.jpg \
  --prompt "What is in the image?"
```

Once `models/modality_projector.py` and `models/vision_language_model.py` are implemented, `generate.py` should run as-is.
