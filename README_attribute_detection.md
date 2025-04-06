# Person Detection and Attribute Classification System

This project implements a multi-attribute person detection system using YOLOv10 for person detection and a custom multi-head classifier for attribute detection. The system can detect and classify various person attributes including gender, age, clothing types, accessories, and actions.

## Features

- Person detection using YOLOv10
- Multi-attribute classification:
  - Gender (Male/Female)
  - Age (Child/Young/Adult/Elderly)
  - Upper clothing (T-shirt/Shirt/Jacket/Coat/Other)
  - Lower clothing (Pants/Shorts/Skirt/Dress/Other)
  - Accessories (Glasses/Hat/Bag/Backpack)
  - Actions (Walking/Standing/Sitting/Running/Other)

## Project Structure

```
├── inference_yolov10.py         # Main inference script for person detection with attributes
├── prepare_attribute_data.py    # Script to prepare training data from P-DESTRE dataset
├── train_attribute_classifier.py # Script to train the attribute classifier
├── yolov10n.pt                  # YOLOv10 nano model weights
├── jpg_Extracted_PIDS/          # Directory containing P-DESTRE images
└── P-DESTRE/                    # P-DESTRE dataset annotations
```

## Implementation Steps

### 1. Data Preparation

The `prepare_attribute_data.py` script extracts person crops from images using bounding boxes and creates attribute labels from P-DESTRE annotations. It organizes the data into train and validation sets.

```bash
python prepare_attribute_data.py
```

This creates an `attribute_data` directory with the following structure:
```
attribute_data/
├── crops/          # All extracted person crops
├── train/          # Training set
├── val/            # Validation set
├── attributes.json # All attribute labels
├── train_attributes.json
└── val_attributes.json
```

### 2. Model Training

The `train_attribute_classifier.py` script trains a multi-head classifier using a pre-trained ResNet18 backbone. Each attribute has a separate classification head.

```bash
python train_attribute_classifier.py
```

The trained models are saved in `attribute_data/models/`.

### 3. Inference

The `inference_yolov10.py` script performs person detection and attribute classification on images:

```bash
python inference_yolov10.py
```

## Model Architecture

### Person Detection

- YOLOv10 nano model for efficient person detection

### Attribute Classification

- Backbone: Pre-trained ResNet18
- Six separate classification heads, one for each attribute group:
  - Gender classification (binary)
  - Age classification (4 classes)
  - Upper clothing classification (5 classes)
  - Lower clothing classification (5 classes)
  - Accessories detection (multi-label, 5 classes)
  - Action classification (5 classes)

## Training Strategy

- Dataset: P-DESTRE dataset with person crops and attribute annotations
- Loss functions:
  - Cross-entropy loss for categorical attributes (gender, age, clothing, actions)
  - Binary cross-entropy loss for multi-label attributes (accessories)
- Optimizer: Adam with learning rate 0.001
- Learning rate scheduling: ReduceLROnPlateau
- Data augmentation: Horizontal flips, color jittering

## Requirements

- Python 3.6+
- PyTorch 1.7+
- TorchVision 0.8+
- Ultralytics 8.0+ (for YOLO)
- OpenCV 4.1+
- NumPy
- Pandas
- Matplotlib
- Pillow

## Future Improvements

1. Implement a unified model (similar to YOLOX-PAI) to directly predict both bounding boxes and attributes
2. Add re-identification features for person tracking
3. Optimize the model for real-time performance
4. Add more attribute categories (e.g., ethnicity, hair color, etc.)
5. Improve accuracy with ensemble methods 