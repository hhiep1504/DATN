import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torchvision.transforms as transforms
from torch import nn
from PIL import Image

# Define attribute labels based on P-DESTRE format
ATTRIBUTE_LABELS = {
    'Gender': ['Male', 'Female', 'Unknown'],
    'Age': [
        'Child (0-11)', 'Teen (12-17)', 'Young (18-24)', 'Young Adult (25-34)',
        'Adult (35-44)', 'Middle Age (45-54)', 'Senior (55-64)', 'Elderly (65+)', 'Unknown'
    ],
    'Upper_Body_Clothing': [
        'T-Shirt', 'Blouse', 'Sweater', 'Coat', 'Bikini', 'Naked', 
        'Dress', 'Uniform', 'Shirt', 'Suit', 'Hoodie', 'Cardigan', 'Unknown'
    ],
    'Lower_Body_Clothing': [
        'Jeans', 'Leggins', 'Pants', 'Shorts', 'Skirt', 
        'Bikini', 'Dress', 'Uniform', 'Suit', 'Unknown'
    ],
    'Accessories': ['No Accessory', 'Has Accessory'],  # Binary classification
    'Action': [
        'Walking', 'Running', 'Standing', 'Sitting', 'Cycling', 
        'Exercising', 'Petting', 'Talking on Phone', 'Leaving Bag', 
        'Fall', 'Fighting', 'Dating', 'Offending', 'Trading'
    ]
}

# Define a multi-attribute classifier model
class AttributeClassifier(nn.Module):
    def __init__(self, num_classes_dict):
        super(AttributeClassifier, self).__init__()
        # Use a pre-trained backbone (e.g., ResNet18)
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add separate heads for each attribute group
        self.gender_head = nn.Linear(512, num_classes_dict['Gender'])
        self.age_head = nn.Linear(512, num_classes_dict['Age'])
        self.upper_body_clothing_head = nn.Linear(512, num_classes_dict['Upper_Body_Clothing'])
        self.lower_body_clothing_head = nn.Linear(512, num_classes_dict['Lower_Body_Clothing'])
        self.accessories_head = nn.Linear(512, 1)  # Binary classification
        self.action_head = nn.Linear(512, num_classes_dict['Action'])
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        return {
            'Gender': self.gender_head(x),
            'Age': self.age_head(x),
            'Upper_Body_Clothing': self.upper_body_clothing_head(x),
            'Lower_Body_Clothing': self.lower_body_clothing_head(x),
            'Accessories': self.accessories_head(x).squeeze(1),  # Binary output
            'Action': self.action_head(x)
        }

def create_attribute_classifier():
    # Count number of classes for each attribute
    num_classes_dict = {
        'Gender': 3,           # Male, Female, Unknown
        'Age': 9,              # 8 age groups + Unknown
        'Upper_Body_Clothing': 13, # 12 clothing types + Unknown
        'Lower_Body_Clothing': 10, # 9 clothing types + Unknown
        'Accessories': 1,      # Binary (has accessory or not)
        'Action': 14           # 13 action types + Unknown
    }
    
    # Create classifier model
    model = AttributeClassifier(num_classes_dict)
    
    # Load pre-trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.load_state_dict(torch.load('attribute_data/models/final_attribute_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def classify_person_attributes(attribute_model, image_path, output_dir, device):
    """Process a single person image (already cropped) and classify attributes"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Define transforms for attribute classification
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL Image and apply transforms
    person_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    person_tensor = transform(person_pil).unsqueeze(0).to(device)
    
    # Classify attributes
    with torch.no_grad():
        attribute_outputs = attribute_model(person_tensor)
    
    # Process each attribute
    attribute_predictions = {}
    
    # Get all predictions first
    for attr_name, output in attribute_outputs.items():
        # Move output to CPU for processing
        output = output.cpu()
        
        # Get prediction
        if attr_name == 'Accessories':
            # For binary output
            pred_idx = 1 if torch.sigmoid(output).item() > 0.5 else 0
        else:
            # For categorical attributes
            pred_idx = torch.argmax(output, dim=1).item()
        
        pred_label = ATTRIBUTE_LABELS[attr_name][pred_idx]
        attribute_predictions[attr_name] = pred_label
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the original image without modifications
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Saved image to: {output_path}")
    
    # Print the attribute information
    print("\nAttribute Analysis Results:")
    print("--------------------------")
    for attr_name, value in attribute_predictions.items():
        print(f"{attr_name}: {value}")
    print("--------------------------")
    
    return attribute_predictions

def find_all_images_in_folder(folder_path):
    """Find all image files in a folder and its subdirectories"""
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    return images

def main():
    # Create attribute classifier and get device
    attribute_model, device = create_attribute_classifier()
    
    # Get image path from user input
    image_path = input("Enter the path to the image: ")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Output directory
    output_dir = 'attribute_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing image: {image_path}")
    
    # Process the specific image
    results = classify_person_attributes(attribute_model, image_path, output_dir, device)

if __name__ == "__main__":
    main() 