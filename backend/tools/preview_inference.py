import os
import glob
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from backend.ml.custom_models import DefectDetectorFasterRCNN


def load_model(ckpt_path: str, num_classes: int = 7) -> DefectDetectorFasterRCNN:
    model = DefectDetectorFasterRCNN(num_classes=num_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def annotate_img(img: Image.Image, dets, color=(255, 0, 0)) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        score = d.get('score', 0.0)
        label = d.get('label', '')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"{label} {score:.2f}"
        tw = max(60, 7 * len(tag))
        draw.rectangle([x1, y1 - 18, x1 + tw, y1], fill=color)
        draw.text((x1 + 4, y1 - 16), tag, fill=(255, 255, 255))
    return img


def make_grid(images, cols=5, bg=(30, 30, 30)) -> Image.Image:
    if not images:
        raise ValueError("No images provided")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new('RGB', (cols * w, rows * h), bg)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def main():
    # Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    val_dir = os.path.join(repo_root, 'backend', 'data', 'val', 'images')
    ckpt = os.path.join(repo_root, 'backend', 'models', 'best_model.pth')
    out_dir = os.path.join(repo_root, 'exports', 'preview')
    os.makedirs(out_dir, exist_ok=True)

    # Gather images
    imgs = sorted(glob.glob(os.path.join(val_dir, '*.jpg')) + glob.glob(os.path.join(val_dir, '*.png')))
    imgs = imgs[:20]
    if not imgs:
        print('No validation images found in', val_dir)
        return

    # Load model
    model = load_model(ckpt)
    to_tensor = transforms.ToTensor()

    annotated = []
    for p in imgs:
        im = Image.open(p).convert('RGB')
        x = to_tensor(im)
        dets = model.detect(x, confidence_threshold=0.3)
        ann = annotate_img(im.copy(), dets)
        base = os.path.basename(p)
        ann.save(os.path.join(out_dir, f'ann_{base}'))
        # normalize size for grid
        annotated.append(ann.resize((512, 512)))

    grid = make_grid(annotated, cols=5)
    grid_path = os.path.join(out_dir, 'preview_grid.jpg')
    grid.save(grid_path)
    print('Wrote:', grid_path)


if __name__ == '__main__':
    main()


