import json
import os
from typing import Dict, List
import shutil

def convert_coco_to_beanscan(coco_file: str, output_dir: str, split: str = 'train'):
    """
    Convert COCO format annotations to BeanScan training format
    
    Args:
        coco_file: Path to COCO JSON file
        output_dir: Output directory for converted data
        split: Dataset split ('train' or 'val')
    """
    
    # Create output directories
    images_dir = os.path.join(output_dir, split, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Convert to BeanScan format
    beanscan_annotations = []
    
    for image_info in coco_data['images']:
        image_id = image_info['id']
        filename = image_info['file_name']
        
        # Get annotations for this image
        annotations = image_annotations.get(image_id, [])
        
        # Convert defects
        defects = []
        for ann in annotations:
            defect = {
                'type': categories.get(ann['category_id'], 'Unknown'),
                'bbox': ann['bbox'],  # COCO format: [x, y, width, height]
                'mask': ann.get('segmentation', [])  # If available
            }
            defects.append(defect)
        
        # Create BeanScan annotation
        beanscan_annotation = {
            'image_id': filename,
            'bean_type': 'Arabica',  # Default, you may need to map this from your data
            'defects': defects,
            'health_score': max(0.1, 1.0 - len(defects) * 0.2)  # Simple health score
        }
        
        beanscan_annotations.append(beanscan_annotation)
    
    # Save converted annotations
    output_file = os.path.join(output_dir, split, f'{split}_annotations.json')
    with open(output_file, 'w') as f:
        json.dump(beanscan_annotations, f, indent=2)
    
    print(f"✅ Converted {len(beanscan_annotations)} annotations for {split} split")
    print(f"📁 Saved to: {output_file}")
    
    return beanscan_annotations

def main():
    """Main conversion function"""
    print("🔄 Converting COCO format to BeanScan format")
    print("=" * 50)
    
    # Configuration
    coco_train_file = r"C:\Users\Raul\Downloads\Coffee Bean Object Detection.v1i.coco\annotations_train.json"  # Split train file
    coco_val_file = r"C:\Users\Raul\Downloads\Coffee Bean Object Detection.v1i.coco\annotations_val.json"     # Split val file
    output_dir = "./data"
    
    # Convert training data
    if os.path.exists(coco_train_file):
        print("📥 Converting training data...")
        convert_coco_to_beanscan(coco_train_file, output_dir, 'train')
    
    # Convert validation data
    if os.path.exists(coco_val_file):
        print("📥 Converting validation data...")
        convert_coco_to_beanscan(coco_val_file, output_dir, 'val')
    
    print("\n🎉 Conversion complete!")
    print("📁 Your data is ready for training in:", output_dir)

if __name__ == "__main__":
    main()
