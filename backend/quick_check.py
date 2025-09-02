import json
import os

def check_bean_distribution():
    """Quick check of bean type distribution"""
    print("🔍 Quick Bean Type Distribution Check")
    print("=" * 40)
    
    # Check train data
    train_file = './data/train/train_annotations.json'
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        bean_counts = {}
        for item in train_data:
            bean_type = item.get('bean_type', 'Unknown')
            bean_counts[bean_type] = bean_counts.get(bean_type, 0) + 1
        
        print(f"📊 Training Data ({len(train_data)} total):")
        for bean_type, count in bean_counts.items():
            pct = (count / len(train_data)) * 100
            print(f"   {bean_type}: {count} ({pct:.1f}%)")
    
    # Check val data
    val_file = './data/val/val_annotations.json'
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        bean_counts = {}
        for item in val_data:
            bean_type = item.get('bean_type', 'Unknown')
            bean_counts[bean_type] = bean_counts.get(bean_type, 0) + 1
        
        print(f"\n📊 Validation Data ({len(val_data)} total):")
        for bean_type, count in bean_counts.items():
            pct = (count / len(val_data)) * 100
            print(f"   {bean_type}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    check_bean_distribution()
