import os
import pandas as pd
from pathlib import Path
import shutil
import cv2  # Added missing import

def get_image_size(image_path):
    """Get image dimensions"""
    img = cv2.imread(str(image_path))  # Convert Path to string
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    return img.shape[1], img.shape[0]  # width, height

def parse_folder_name(folder_name):
    """Parse the folder name into its components"""
    parts = folder_name.split('-')
    if len(parts) == 5:
        return {
            'day': parts[0],
            'month': parts[1],
            'year': parts[2],
            'camera': parts[3],
            'sequence': parts[4],
            'date_str': f"{parts[0]}{parts[1]}{parts[2]}"
        }
    return None

def create_yolo_labels(annotation_file, base_image_dir, output_dir):
    """
    Convert PReID dataset labels to YOLO format
    Args:
        annotation_file: Path to the annotation file
        base_image_dir: Base directory containing date folders with ID subfolders
        output_dir: Directory to save YOLO format labels
    """
    print(f"\nProcessing annotation file: {annotation_file}")
    
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
    
    # Get date folder name from annotation file
    date_folder = annotation_file.stem  # e.g., "08-11-2019-1-1"
    folder_info = parse_folder_name(date_folder)
    
    if folder_info:
        print(f"\nFolder structure breakdown:")
        print(f"Date: {folder_info['day']}/{folder_info['month']}/{folder_info['year']}")
        print(f"Camera: {folder_info['camera']}")
        print(f"Sequence: {folder_info['sequence']}")
        print(f"Date string for filenames: {folder_info['date_str']}")
    else:
        print("Warning: Unexpected folder name format")
        return 0, 0
    
    # Process each annotation
    processed_count = 0
    skipped_count = 0
    total_views_processed = 0
    
    print("\nProcessing annotations...")
    for idx, row in df.iterrows():
        frame = int(row[0])
        id_ = int(row[1])
        x = float(row[2])
        y = float(row[3])
        h = float(row[4])
        w = float(row[5])
        
        # Get all attributes
        attributes = {
            'gender': int(row[10]),
            'age': int(row[11]),
            'height': int(row[12]),
            'body_volume': int(row[13]),
            'ethnicity': int(row[14]),
            'hair_color': int(row[15]),
            'hairstyle': int(row[16]),
            'beard': int(row[17]),
            'moustache': int(row[18]),
            'glasses': int(row[19]),
            'head_accessories': int(row[20]),
            'clothing_top': int(row[21]),
            'clothing_bottom': int(row[22]),
            'feet': int(row[23]),
            'accessories': int(row[24]),
            'action': int(row[25])
        }
        
        # Expected image path base
        id_folder = Path(base_image_dir) / date_folder / str(id_)
        
        # Check if folder exists
        if not id_folder.exists():
            print(f"Warning: ID folder not found: {id_folder}")
            skipped_count += 1
            continue
        
        # Try each possible view ID (0-74)
        views_found = []
        for view_id in range(75):  # 0 to 74
            image_name = f"{id_}_{frame}_{folder_info['date_str']}_{view_id}.jpg"
            image_path = id_folder / image_name
            
            if image_path.exists():
                views_found.append(view_id)
                img_width, img_height = get_image_size(image_path)
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Determine train/val split (90% train, 10% val)
                split = 'train' if frame % 10 != 0 else 'val'
                
                # Create label file using the same image name
                output_path = os.path.join(output_dir, 'labels', split, image_path.name.replace('.jpg', '.txt'))
                
                # Write YOLO format label with all attributes
                with open(output_path, 'w') as f:  # Changed from 'a' to 'w' to avoid duplicates
                    # Write detection box and all attributes
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    for attr_value in attributes.values():
                        f.write(f" {attr_value}")
                    f.write("\n")
                
                # Copy image to YOLO directory structure
                dst_img_path = os.path.join(output_dir, 'images', split, image_path.name)
                shutil.copy2(str(image_path), dst_img_path)
                
                total_views_processed += 1
        
        if views_found:
            print(f"✓ Processed {len(views_found)} views for person {id_} frame {frame}: {views_found}")
            processed_count += 1
        else:
            print(f"Warning: No views found for person {id_} in frame {frame}")
            skipped_count += 1
    
    print(f"\nProcessing Summary:")
    print(f"Annotations processed: {processed_count}")
    print(f"Annotations skipped: {skipped_count}")
    print(f"Total views processed: {total_views_processed}")
    if processed_count > 0:
        print(f"Average views per annotation: {total_views_processed/processed_count:.1f}")
    
    return processed_count, skipped_count

def create_data_yaml(output_dir):
    """Create data.yaml file with all P-DESTRE attributes"""
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

age_names:
  0: 0-11
  1: 12-17
  2: 18-24
  3: 25-34
  4: 35-44
  5: 45-54
  6: 55-64
  7: >65
  8: unknown

height_names:
  0: child
  1: short
  2: medium
  3: tall
  4: unknown

body_volume_names:
  0: thin
  1: medium
  2: fat
  3: unknown

ethnicity_names:
  0: white
  1: black
  2: asian
  3: indian
  4: unknown

hair_color_names:
  0: black
  1: brown
  2: white
  3: red
  4: gray
  5: occluded
  6: unknown

hairstyle_names:
  0: bald
  1: short
  2: medium
  3: long
  4: horse_tail
  5: unknown

beard_names:
  0: yes
  1: no
  2: unknown

moustache_names:
  0: yes
  1: no
  2: unknown

glasses_names:
  0: normal_glass
  1: sun_glass
  2: no
  3: unknown

head_accessories_names:
  0: hat
  1: scarf
  2: neckless
  3: cannot_see
  4: unknown

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

feet_names:
  0: sport_shoe
  1: classic_shoe
  2: high_heels
  3: boots
  4: sandal
  5: nothing
  6: unknown

accessories_names:
  0: bag
  1: backpack_bag
  2: rolling_bag
  3: umbrella
  4: sport_bag
  5: market_bag
  6: nothing
  7: unknown

action_names:
  0: walking
  1: running
  2: standing
  3: sitting
  4: cycling
  5: exercising
  6: petting
  7: talking_over_phone
  8: leaving_bag
  9: fall
  10: fighting
  11: dating
  12: offending
  13: trading
""")

def main():
    # Define paths
    annotation_dir = Path("P-DESTRE/annotation")
    base_image_dir = "jpg_Extracted_PIDS"
    output_dir = "dataset_yolo"
    
    # Get all annotation files
    annotation_files = [f for f in annotation_dir.glob("*.txt") 
                       if not f.name.startswith("._") and not f.name.startswith(".")]
    
    if not annotation_files:
        print("No annotation files found!")
        return
    
    # Sort files for consistent processing
    annotation_files.sort()
    
    # Process only the first annotation file (index 0)
    first_file = annotation_files[0]  # Changed from index 3 to index 0
    print(f"\nProcessing first annotation file: {first_file.name}")
    
    try:
        processed, skipped = create_yolo_labels(first_file, base_image_dir, output_dir)
        if processed == 0:
            print("No annotations were processed successfully.")
            return
    except Exception as e:
        print(f"Error processing {first_file.name}: {e}")
        return
    
    # Create data.yaml after processing
    create_data_yaml(output_dir)
    print(f"\nSuccessfully converted labels to YOLO format in {output_dir}")

if __name__ == "__main__":
    main()