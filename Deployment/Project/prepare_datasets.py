"""Admin script: download datasets once and save them to a shared filesystem.

Usage (run as an admin or any user with write access to SHARED_PATH):
    python prepare_datasets.py --shared_path /your/shared/path

After this script finishes, set in TrainConfig or via CLI:
    --dataset_local_path /your/shared/path/the_cauldron/ai2d   (for Cauldron)
    --dataset_local_path /your/shared/path/flickr30k           (for Flickr)
    --dataset_local_path /your/shared/path/mmstar              (for MMStar eval)

Then make the directory world-readable:
    chmod -R a+rX /your/shared/path
"""

import argparse
import os


CAULDRON_SUBSETS = [
    "ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "clevr_math",
    "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa",
    "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam",
    "iconqa", "infographic_vqa", "intergps", "localized_narratives",
    "mapqa", "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa", "okvqa",
    "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql",
    "robut_wtq", "scienceqa", "screen2words", "spot_the_diff", "st_vqa",
    "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa",
    "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr",
    "websight",
]
FLICKR_REPO      = "AnyModal/flickr30k"
CAULDRON_REPO    = "HuggingFaceM4/the_cauldron"
MMSTAR_REPO      = "Lin-Chen/MMStar"


def dataset_exists(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "dataset_info.json"))
        or os.path.exists(os.path.join(path, "dataset_dict.json"))
    )


def save_cauldron(shared_path: str, subsets: list[str]):
    from datasets import load_dataset

    failed = []
    for subset in subsets:
        out_dir = os.path.join(shared_path, "the_cauldron", subset)
        if dataset_exists(out_dir):
            print(f"  [skip] {subset} already exists at {out_dir}")
            continue
        print(f"  Downloading the_cauldron/{subset} …")
        try:
            ds = load_dataset(CAULDRON_REPO, subset)
            ds.save_to_disk(out_dir)
            print(f"  Saved to {out_dir}")
        except Exception as e:
            print(f"  [FAILED] {subset}: {e}")
            failed.append(subset)
    if failed:
        print(f"\n  Warning: {len(failed)} subset(s) failed: {failed}")


def save_flickr(shared_path: str):
    from datasets import load_dataset

    out_dir = os.path.join(shared_path, "flickr30k")
    if dataset_exists(out_dir):
        print(f"  [skip] flickr30k already exists at {out_dir}")
        return
    print("  Downloading flickr30k …")
    ds = load_dataset(FLICKR_REPO)
    ds.save_to_disk(out_dir)
    print(f"  Saved to {out_dir}")


def save_mmstar(shared_path: str):
    from datasets import load_dataset

    out_dir = os.path.join(shared_path, "mmstar")
    if dataset_exists(out_dir):
        print(f"  [skip] MMStar already exists at {out_dir}")
        return
    print("  Downloading MMStar …")
    ds = load_dataset(MMSTAR_REPO)
    ds.save_to_disk(out_dir)
    print(f"  Saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shared_path", type=str, required=True,
                   help="Shared directory where datasets will be saved.")
    p.add_argument("--skip_cauldron", action="store_true")
    p.add_argument("--skip_flickr", action="store_true")
    p.add_argument("--skip_mmstar", action="store_true")
    p.add_argument("--cauldron_subsets", type=str, default=",".join(CAULDRON_SUBSETS),
                   help="Comma-separated list of Cauldron subsets to download.")
    args = p.parse_args()

    os.makedirs(args.shared_path, exist_ok=True)

    if not args.skip_cauldron:
        print("=== The Cauldron ===")
        save_cauldron(args.shared_path, args.cauldron_subsets.split(","))

    if not args.skip_flickr:
        print("=== Flickr30k ===")
        save_flickr(args.shared_path)

    if not args.skip_mmstar:
        print("=== MMStar ===")
        save_mmstar(args.shared_path)

    print(f"\nDone. Make the directory readable by all users:")
    print(f"    chmod -R a+rX {args.shared_path}")
    print(f"\nThen train with:")
    print(f"    python train.py --dataset_local_path {args.shared_path}/the_cauldron/ai2d")
    print(f"  or")
    print(f"    python train.py --dataset_type flickr --dataset_local_path {args.shared_path}/flickr30k")
    print(f"\nThen evaluate with:")
    print(f"    python eval_mmstar.py --checkpoint checkpoints/best_step5000 --dataset_local_path {args.shared_path}/mmstar --split val")


if __name__ == "__main__":
    main()
