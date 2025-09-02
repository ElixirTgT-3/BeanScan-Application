import json
import os
import random
import shutil
from typing import List, Dict

def split_dataset(annotations_file: str, train_ratio: float = 0.9, random_seed: int = 42):
    """
    Split dataset into train and validation sets
    
    Args:
        annotations_file: Path to COCO annotations file
        train_ratio: Ratio of data to use for training (default: 0.9 = 90%)
        random_seed: Random seed for reproducible splits
    """
    
    # Set random seed
    random.seed(random_seed)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Shuffle and split
    random.shuffle(image_ids)
    split_index = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_index])
    val_ids = set(image_ids[split_index:])
    
    # Create train and val datasets
    train_data = {
        'images': [],
        'annotations': [],
        'categories': coco_data['categories']
    }
    
    val_data = {
        'images': [],
        'annotations': [],
        'categories': coco_data['categories']
    }
    
    # Split images
    for img in coco_data['images']:
        if img['id'] in train_ids:
            train_data['images'].append(img)
        else:
            val_data['images'].append(img)
    
    # Split annotations
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)
    
    # Save split datasets
    train_file = annotations_file.replace('.json', '_train.json')
    val_file = annotations_file.replace('.json', '_val.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ Dataset split complete!")
    print(f"📊 Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"📊 Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    print(f"📁 Train file: {train_file}")
    print(f"📁 Val file: {val_file}")
    
    return train_file, val_file

def copy_images_to_split_folders(images_dir: str, train_file: str, val_file: str):
    """
    Copy images to appropriate train/val folders based on split
    
    Args:
        images_dir: Directory containing all images
        train_file: Path to train annotations file
        val_file: Path to val annotations file
    """
    
    # Load train and val image lists
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    
    train_images = {img['file_name'] for img in train_data['images']}
    val_images = {img['file_name'] for img in val_data['images']}
    
    # Create directories
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/val/images', exist_ok=True)
    
    # Copy images
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(images_dir, filename)
            
            if filename in train_images:
                dst_path = os.path.join('data/train/images', filename)
                shutil.copy2(src_path, dst_path)
            elif filename in val_images:
                dst_path = os.path.join('data/val/images', filename)
                shutil.copy2(src_path, dst_path)
    
    print(f"✅ Images copied to train/val folders")

def main():
    """Main function to split dataset and organize files"""
    print("🔄 Splitting dataset into train/validation sets")
    print("=" * 50)
    
    # Configuration - UPDATE THESE PATHS
    annotations_file = r"C:\Users\Raul\Downloads\Coffee Bean Object Detection.v1i.coco\annotations.json"  # Your extracted annotations file
    images_dir = r"C:\Users\Raul\Downloads\Coffee Bean Object Detection.v1i.coco\images"                 # Your extracted images folder
    
    if not os.path.exists(annotations_file):
        print(f"❌ Annotations file not found: {annotations_file}")
        print("Please update the path in the script")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("Please update the path in the script")
        return
    
    # Split dataset
    train_file, val_file = split_dataset(annotations_file, train_ratio=0.9)
    
    # Copy images to appropriate folders
    copy_images_to_split_folders(images_dir, train_file, val_file)
    
    print("\n🎉 Dataset organization complete!")
    print("📁 Your data is ready for conversion and training")

if __name__ == "__main__":
    main()
