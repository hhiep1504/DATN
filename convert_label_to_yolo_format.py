import pandas as pd
import os
from pathlib import Path
import cv2

def get_image_size(image_path):
    """Get image dimensions"""
    img = cv2.imread(str(image_path))  # Convert Path to string
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    return img.shape[1], img.shape[0]  # width, height

def create_yolo_labels(annotation_file, base_image_dir, output_dir):
    """
    Convert PReID dataset labels to YOLO format
    Args:
        annotation_file: Path to the annotation file
        base_image_dir: Base directory containing ID folders with images
        output_dir: Directory to save YOLO format labels
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'labels', 'train')
    val_dir = os.path.join(output_dir, 'labels', 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create image directories for YOLO
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    # Read annotations with comma delimiter
    df = pd.read_csv(annotation_file, header=None, delimiter=',')
    
    # Filter out untracked persons (ID = -1)
    df = df[df[1] != -1].copy()
    print(f"Total annotations after filtering out untracked persons: {len(df)}")
    
    # Define attribute mappings
    gender_map = {0: 0, 1: 1, 2: 2}  # 0=Nam, 1=Nữ, 2=Không xác định
    clothing_top_map = {
        0: 0,  # T Shirt
        1: 1,  # Blouse
        2: 2,  # Sweater
        3: 3,  # Coat
        4: 4,  # Bikini
        5: 5,  # Naked
        6: 6,  # Dress
        7: 7,  # Uniform
        8: 8,  # Shirt
        9: 9,  # Suit
        10: 10,  # Hoodie
        11: 11,  # Cardigan
        12: 12   # Unknown
    }
    clothing_bottom_map = {
        0: 0,  # Jeans
        1: 1,  # Leggins
        2: 2,  # Pants
        3: 3,  # Shorts
        4: 4,  # Skirt
        5: 5,  # Bikini
        6: 6,  # Dress
        7: 7,  # Uniform
        8: 8,  # Suit
        9: 9   # Unknown
    }
    
    # Process each annotation
    processed_count = 0
    skipped_count = 0
    
    for _, row in df.iterrows():
        frame = int(row[0])  # Convert frame to integer
        id_ = int(row[1])  # Person ID
        x = float(row[2])  # x coordinate
        y = float(row[3])  # y coordinate
        h = float(row[4])  # height
        w = float(row[5])  # width
        gender = int(row[10])  # gender
        clothing_top = int(row[20])  # upper clothing
        clothing_bottom = int(row[21])  # lower clothing
        
        # Get image path with correct format
        image_name = f"{id_}_{frame}_08112019_0.jpg"
        image_path = Path(base_image_dir) / str(id_) / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            skipped_count += 1
            continue
            
        img_width, img_height = get_image_size(image_path)
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        # Map attributes to indices
        gender_idx = gender_map.get(gender, 2)  # Default to unknown
        top_idx = clothing_top_map.get(clothing_top, 12)  # Default to unknown
        bottom_idx = clothing_bottom_map.get(clothing_bottom, 9)  # Default to unknown
        
        # Determine train/val split (90% train, 10% val)
        split = 'train' if frame % 10 != 0 else 'val'
        
        # Create label file
        output_path = os.path.join(output_dir, 'labels', split, image_name.replace('.jpg', '.txt'))
        
        # Write YOLO format label
        with open(output_path, 'a') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {gender_idx} {top_idx} {bottom_idx}\n")
            
        # Copy image to YOLO directory structure
        import shutil
        dst_img_path = os.path.join(output_dir, 'images', split, image_name)
        shutil.copy2(str(image_path), dst_img_path)
        
        processed_count += 1
    
    print(f"\nProcessing Summary:")
    print(f"Total annotations processed: {processed_count}")
    print(f"Total annotations skipped: {skipped_count}")
    
    # Create data.yaml file
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: person

# Number of classes
nc: 1

# Attribute classes
gender_names:
  0: male
  1: female
  2: unknown

clothing_top_names:
  0: tshirt
  1: blouse
  2: sweater
  3: coat
  4: bikini
  5: naked
  6: dress
  7: uniform
  8: shirt
  9: suit
  10: hoodie
  11: cardigan
  12: unknown

clothing_bottom_names:
  0: jeans
  1: leggins
  2: pants
  3: shorts
  4: skirt
  5: bikini
  6: dress
  7: uniform
  8: suit
  9: unknown
""")

def main():
    # Define paths
    annotation_file = "P-DESTRE/annotation/08-11-2019-1-1.txt"  # Update with your annotation file path
    base_image_dir = "jpg_Extracted_PIDS/08-11-2019-1-1"  # Base directory containing ID folders
    output_dir = "dataset_yolo"  # Output directory for YOLO format
    
    try:
        create_yolo_labels(annotation_file, base_image_dir, output_dir)
        print(f"Successfully converted labels to YOLO format in {output_dir}")
    except Exception as e:
        print(f"Error converting labels: {e}")

if __name__ == "__main__":
    main()