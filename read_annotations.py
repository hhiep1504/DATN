import pandas as pd
import os
from pathlib import Path

# Định nghĩa ý nghĩa các cột
COLUMN_MEANINGS = {
    0: "Frame (Số frame trong video)",
    1: "ID (ID của người)",
    2: "x (Tọa độ x của bounding box)",
    3: "y (Tọa độ y của bounding box)",
    4: "h (Chiều cao của bounding box)",
    5: "w (Chiều rộng của bounding box)",
    6: "flag (Trạng thái head pose: -1=Not available, 1=Available)",
    7: "yaw (Góc quay đầu theo trục y - độ)",
    8: "pitch (Góc quay đầu theo trục x - độ)",
    9: "roll (Góc quay đầu theo trục z - độ)",
    10: "Gender (0=Nam, 1=Nữ, 2=Không xác định)",
    11: "Age (0=0-11, 1=12-17, 2=18-24, 3=25-34, 4=35-44, 5=45-54, 6=55-64, 7=>65, 8=Unknown)",
    12: "Height (0=Child, 1=Short, 2=Medium, 3=Tall, 4=Unknown)",
    13: "Body Volume (0=Thin, 1=Medium, 2=Fat, 3=Unknown)",
    14: "Ethnicity (0=White, 1=Black, 2=Asian, 3=Indian, 4=Unknown)",
    15: "Hair Color (0=Black, 1=Brown, 2=White, 3=Red, 4=Gray, 5=Occluded, 6=Unknown)",
    16: "Hairstyle (0=Bald, 1=Short, 2=Medium, 3=Long, 4=Horse Tail, 5=Unknown)",
    17: "Beard (0=Yes, 1=No, 2=Unknown)",
    18: "Moustache (0=Yes, 1=No, 2=Unknown)",
    19: "Glasses (0=Normal glass, 1=Sun glass, 2=No, 3=Unknown)",
    20: "Head Accessories (0=Hat, 1=Scarf, 2=Neckless, 3=Cannot see, 4=Unknown)",
    21: "Upper Body Clothing (0=T Shirt, 1=Blouse, 2=Sweater, 3=Coat, 4=Bikini, 5=Naked, 6=Dress, 7=Uniform, 8=Shirt, 9=Suit, 10=Hoodie, 11=Cardigan, 12=Unknown)",
    22: "Lower Body Clothing (0=Jeans, 1=Leggins, 2=Pants, 3=Shorts, 4=Skirt, 5=Bikini, 6=Dress, 7=Uniform, 8=Suit, 9=Unknown)",
    23: "Feet (0=Sport Shoe, 1=Classic Shoe, 2=High Heels, 3=Boots, 4=Sandal, 5=Nothing, 6=Unknown)",
    24: "Accessories (0=Bag, 1=Backpack Bag, 2=Rolling Bag, 3=Umbrella, 4=Sport Bag, 5=Market Bag, 6=Nothing, 7=Unknown)",
    25: "Action (0=Walking, 1=Running, 2=Standing, 3=Sitting, 4=Cycling, 5=Exercising, 6=Petting, 7=Talking over Phone, 8=Leaving Bag, 9=Fall, 10=Fighting, 11=Dating, 12=Offending, 13=Trading)"
}

def read_large_annotation(file_path, chunk_size=1000):
    """
    Đọc file annotation lớn theo từng phần nhỏ và hiển thị ý nghĩa của các cột
    Args:
        file_path: đường dẫn đến file annotation
        chunk_size: số dòng đọc mỗi lần
    """
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            chunks = pd.read_csv(file_path, chunksize=chunk_size, header=None, encoding=encoding)
            
            for i, chunk in enumerate(chunks):
                print(f"\nDòng đầu tiên của file {file_path.name}:")
                first_row = chunk.iloc[0]
                for col, value in first_row.items():
                    print(f"{COLUMN_MEANINGS[col]}: {value}")
                
                if i == 0:
                    break
                    
            print(f"\nĐọc file thành công với encoding: {encoding}")
            return
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Lỗi khác khi đọc file với encoding {encoding}: {e}")
            continue
    
    print("Không thể đọc file với bất kỳ encoding nào!")

def main():
    annotation_dir = Path("P-DESTRE/annotation")
    
    annotation_files = [f for f in annotation_dir.glob("*.txt") 
                       if not f.name.startswith("._") and not f.name.startswith(".")]
    
    if not annotation_files:
        print("Không tìm thấy file annotation!")
        return
    
    smallest_file = min(annotation_files, key=lambda x: x.stat().st_size)
    print(f"Đang đọc file: {smallest_file.name}")
    print(f"Kích thước file: {smallest_file.stat().st_size / 1024:.2f} KB")
    
    try:
        read_large_annotation(smallest_file)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    main() 