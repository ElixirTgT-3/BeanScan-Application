import json
import os
import shutil
from typing import Dict, List

def convert_coco_to_beanscan_direct(coco_file: str, output_dir: str, split: str = 'train'):
    """
    Convert COCO format annotations directly to BeanScan training format
    
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
        
        # Extract bean type from filename
        bean_type = 'Arabica'  # Default
        if filename.startswith('Arabica_'):
            bean_type = 'Arabica'
        elif filename.startswith('Robusta_'):
            bean_type = 'Robusta'
        elif filename.startswith('Liberica_'):
            bean_type = 'Liberica'
        elif filename.startswith('Excelsa_'):
            bean_type = 'Excelsa'
        else:
            bean_type = 'Arabica'  # Default to Arabica instead of Other
        
        # Create BeanScan annotation
        beanscan_annotation = {
            'image_id': filename,
            'bean_type': bean_type,
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

def copy_images_from_folder(source_folder: str, target_folder: str):
    """
    Copy all images from source folder to target folder
    
    Args:
        source_folder: Source folder containing images
        target_folder: Target folder to copy images to
    """
    
    # Create target directory
    os.makedirs(target_folder, exist_ok=True)
    
    # Copy all image files
    copied_count = 0
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(target_folder, filename)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    
    print(f"✅ Copied {copied_count} images to {target_folder}")

def main():
    """Main function to process the already-split dataset"""
    print("🔄 Processing Coffee Bean Object Detection Dataset")
    print("=" * 60)
    
    # Configuration - Your extracted dataset paths
    dataset_root = r"C:\Users\Raul\Downloads\Second Set Coffee Bean Object Detection.v1i.coco (1)"
    train_folder = os.path.join(dataset_root, "train")
    valid_folder = os.path.join(dataset_root, "valid")
    
    # COCO annotation files
    train_annotations = os.path.join(train_folder, "_annotations.coco.json")
    valid_annotations = os.path.join(valid_folder, "_annotations.coco.json")
    
    # Output directory
    output_dir = "./data"
    
    # Check if files exist
    if not os.path.exists(train_annotations):
        print(f"❌ Train annotations not found: {train_annotations}")
        return
    
    if not os.path.exists(valid_annotations):
        print(f"❌ Valid annotations not found: {valid_annotations}")
        return
    
    print("📊 Processing training data...")
    # Convert train annotations
    convert_coco_to_beanscan_direct(train_annotations, output_dir, 'train')
    
    # Copy train images
    train_images_target = os.path.join(output_dir, 'train', 'images')
    copy_images_from_folder(train_folder, train_images_target)
    
    print("\n📊 Processing validation data...")
    # Convert valid annotations
    convert_coco_to_beanscan_direct(valid_annotations, output_dir, 'val')
    
    # Copy valid images
    valid_images_target = os.path.join(output_dir, 'val', 'images')
    copy_images_from_folder(valid_folder, valid_images_target)
    
    print("\n🎉 Dataset processing complete!")
    print("📁 Your data is ready for training in:", output_dir)
    
    # Show final structure
    print("\n📂 Final data structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/ (training images)")
    print(f"  │   └── train_annotations.json")
    print(f"  └── val/")
    print(f"      ├── images/ (validation images)")
    print(f"      └── val_annotations.json")

if __name__ == "__main__":
    main()
