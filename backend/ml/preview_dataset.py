import os
import json
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = 'backend/data'
EXPORT_DIR = 'exports/dataset_preview'

DEFECT_CLASS_NAMES = [
    "insect_damage",
    "nugget",
    "quaker",
    "roasted-beans",
    "shell",
    "under_roast",
]


def load_annotations(split: str) -> List[Dict]:
    ann_path = os.path.join(DATA_ROOT, split, f'{split}_annotations.json')
    with open(ann_path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_bbox(image: Image.Image, bbox, label: str, color=(255, 0, 0)) -> Image.Image:
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    text = label
    draw.rectangle([x, y - 18, x + 6 * max(6, len(text)), y], fill=color)
    draw.text((x + 4, y - 16), text, fill=(255, 255, 255))
    return image


def summarize_and_preview(split: str, max_images: int = 24):
    anns = load_annotations(split)

    # Class counts
    counts = {k: 0 for k in DEFECT_CLASS_NAMES}

    out_dir = os.path.join(EXPORT_DIR, split)
    ensure_dir(out_dir)

    previewed = 0
    for ann in anns:
        img_path = os.path.join(DATA_ROOT, split, 'images', ann['image_id'])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert('RGB')
        for d in ann.get('defects', []):
            cls = str(d.get('type', '')).strip()
            if cls in counts:
                counts[cls] += 1
            bbox = d.get('bbox')
            if bbox and len(bbox) == 4:
                img = draw_bbox(img, bbox, cls)

        if previewed < max_images:
            img.save(os.path.join(out_dir, ann['image_id']))
            previewed += 1

    # Save stats
    stats_path = os.path.join(EXPORT_DIR, f'{split}_class_counts.json')
    with open(stats_path, 'w') as f:
        json.dump(counts, f, indent=2)

    print(f"Saved {previewed} preview images to {out_dir}")
    print(f"Class counts written to {stats_path}")


def main():
    ensure_dir(EXPORT_DIR)
    for split in ['train', 'val']:
        try:
            summarize_and_preview(split)
        except FileNotFoundError:
            print(f"No annotations found for split: {split}")


if __name__ == '__main__':
    main()


