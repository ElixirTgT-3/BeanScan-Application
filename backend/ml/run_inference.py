import os
import sys
import random
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms
from PIL import ImageDraw

from .custom_models import DefectDetectorFasterRCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='backend/models/defect_frcnn_best.pth')
    parser.add_argument('--fallback', default='backend/models/defect_frcnn_epoch_1.pth')
    parser.add_argument('--image', default='')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    ckpt = args.ckpt if os.path.exists(args.ckpt) else args.fallback
    if not os.path.exists(ckpt):
        print('Checkpoint not found:', ckpt)
        sys.exit(1)

    if args.image and os.path.exists(args.image):
        img_path = args.image
    else:
        candidates = glob.glob('backend/data/val/images/*.jpg') or glob.glob('backend/data/train/images/*.jpg')
        if not candidates:
            print('No images found under backend/data')
            sys.exit(1)
        img_path = random.choice(candidates)

    print('Using checkpoint:', ckpt)
    print('Image:', img_path)

    model = DefectDetectorFasterRCNN(num_classes=6, pretrained=False)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

    img = Image.open(img_path).convert('RGB')
    x = transforms.ToTensor()(img)

    detections = model.detect(x, confidence_threshold=args.threshold)
    print('Detections (top 5):')
    for d in detections[:5]:
        print(d)

    # Save annotated image
    if detections:
        draw = ImageDraw.Draw(img)
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            label = f"{d.get('label','')} {d.get('score',0):.2f}"
            draw.rectangle([x1, y1 - 18, x1 + max(60, len(label)*7), y1], fill=(255,0,0))
            draw.text((x1 + 4, y1 - 16), label, fill=(255,255,255))
        os.makedirs('exports', exist_ok=True)
        out_path = os.path.join('exports', 'inference_preview.png')
        img.save(out_path)
        print('Saved annotated preview to:', out_path)


if __name__ == '__main__':
    main()


