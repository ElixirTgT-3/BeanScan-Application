import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco_annotations(coco_path: Path) -> Dict:
    with coco_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_image_label_map(
    coco_data: Dict,
    target_classes: List[str],
) -> Dict[int, str]:
    """Map image id to bean label derived from COCO annotations."""
    category_lookup = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    # Filter to ids of desired classes
    target_ids = {cat_id for cat_id, name in category_lookup.items() if name in target_classes}

    image_labels: Dict[int, List[str]] = {}
    for ann in coco_data.get("annotations", []):
        category_id = ann.get("category_id")
        if category_id not in target_ids:
            continue
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        image_labels.setdefault(image_id, []).append(category_lookup[category_id])

    resolved_labels: Dict[int, str] = {}
    for image_id, labels in image_labels.items():
        if not labels:
            continue
        # Select the most frequent label for this image; tie broken deterministically
        label_counts: Dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        sorted_labels = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        resolved_labels[image_id] = sorted_labels[0][0]

    return resolved_labels


def split_dataset(image_ids: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    val_ids = set(shuffled[:val_count])
    train_ids = [img_id for img_id in shuffled if img_id not in val_ids]
    return train_ids, list(val_ids)


def write_split(
    split_name: str,
    image_ids: List[int],
    coco_data: Dict,
    image_to_label: Dict[int, str],
    images_dir: Path,
    output_dir: Path,
) -> None:
    split_dir = output_dir / split_name
    images_output = split_dir / "images"
    split_dir.mkdir(parents=True, exist_ok=True)
    images_output.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict] = []

    image_lookup = {img["id"]: img for img in coco_data.get("images", [])}

    for img_id in image_ids:
        image_meta = image_lookup.get(img_id)
        if not image_meta:
            continue
        label = image_to_label.get(img_id)
        if label is None:
            continue
        src_filename = image_meta.get("file_name")
        if not src_filename:
            continue

        source_path = images_dir / src_filename
        if not source_path.exists():
            print(f"[WARN] Missing source image: {source_path}")
            continue

        dest_filename = Path(src_filename).name
        dest_path = images_output / dest_filename
        shutil.copy2(source_path, dest_path)

        annotations.append(
            {
                "image_id": dest_filename,
                "bean_type": label,
                "health_score": 1.0,
            }
        )

    annotations_path = split_dir / f"{split_name}_annotations.json"
    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
    print(f"[OK] Wrote {len(annotations)} annotations to {annotations_path}")


def convert_coco_to_classification(
    coco_path: Path,
    images_dir: Path,
    output_dir: Path,
    target_classes: List[str],
    val_ratio: float,
    seed: int,
) -> None:
    coco_data = load_coco_annotations(coco_path)
    image_to_label = prepare_image_label_map(coco_data, target_classes)

    if not image_to_label:
        raise RuntimeError("No images matched the requested target classes.")

    image_ids = list(image_to_label.keys())
    train_ids, val_ids = split_dataset(image_ids, val_ratio, seed)

    if not train_ids:
        raise RuntimeError("Training split is empty after conversion.")

    output_dir.mkdir(parents=True, exist_ok=True)

    write_split("train", train_ids, coco_data, image_to_label, images_dir, output_dir)
    write_split("val", val_ids, coco_data, image_to_label, images_dir, output_dir)

    print(f"[DONE] Dataset ready in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO annotations to BeanScan classification format.")
    parser.add_argument("--coco-json", required=True, type=Path, help="Path to the source COCO annotation file.")
    parser.add_argument("--images-dir", required=True, type=Path, help="Directory containing the source images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for the classification dataset.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Liberica", "Excelsa"],
        help="Target bean class names to keep.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Portion of data to use for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_coco_to_classification(
        coco_path=Path(args.coco_json),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        target_classes=args.classes,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
