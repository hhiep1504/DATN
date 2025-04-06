import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Define the same attribute classifier model
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

# Custom dataset for attribute data
class AttributeDataset(Dataset):
    def __init__(self, data_dir, attribute_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load attribute data
        with open(attribute_file, 'r') as f:
            self.attribute_data = json.load(f)
        
        self.image_files = list(self.attribute_data.keys())
        
        # Define attribute mappings (string to index)
        self.attribute_maps = {
            'Gender': {'Male': 0, 'Female': 1, 'Unknown': 2},
            'Age': {
                'Child (0-11)': 0, 
                'Teen (12-17)': 1, 
                'Young (18-24)': 2, 
                'Young Adult (25-34)': 3,
                'Adult (35-44)': 4,
                'Middle Age (45-54)': 5,
                'Senior (55-64)': 6,
                'Elderly (65+)': 7,
                'Unknown': 8
            },
            'Upper_Body_Clothing': {
                'T-Shirt': 0, 'Blouse': 1, 'Sweater': 2, 'Coat': 3, 'Bikini': 4,
                'Naked': 5, 'Dress': 6, 'Uniform': 7, 'Shirt': 8, 'Suit': 9,
                'Hoodie': 10, 'Cardigan': 11, 'Unknown': 12
            },
            'Lower_Body_Clothing': {
                'Jeans': 0, 'Leggins': 1, 'Pants': 2, 'Shorts': 3, 'Skirt': 4,
                'Bikini': 5, 'Dress': 6, 'Uniform': 7, 'Suit': 8, 'Unknown': 9
            },
            'Accessories': {
                'Bag': 0, 'Backpack': 1, 'Rolling Bag': 2, 'Umbrella': 3,
                'Sport Bag': 4, 'Market Bag': 5, 'Nothing': 6, 'Unknown': 7
            },
            'Action': {
                'Walking': 0, 'Running': 1, 'Standing': 2, 'Sitting': 3, 'Cycling': 4,
                'Exercising': 5, 'Petting': 6, 'Talking on Phone': 7, 'Leaving Bag': 8,
                'Fall': 9, 'Fighting': 10, 'Dating': 11, 'Offending': 12, 'Trading': 13
            }
        }
        
        # For multi-label attributes like accessories, we'll handle them differently
        self.accessory_types = ['Glasses', 'Hat', 'Bag', 'Backpack', 'None']
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image file and path
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get attributes for this image
        attr_data = self.attribute_data[img_file]
        
        # Convert attributes to tensors
        labels = {
            'Gender': torch.tensor(self.attribute_maps['Gender'].get(attr_data.get('Gender', 'Unknown'), 2), dtype=torch.long),
            'Age': torch.tensor(self.attribute_maps['Age'].get(attr_data.get('Age', 'Unknown'), 8), dtype=torch.long),
            'Upper_Body_Clothing': torch.tensor(self.attribute_maps['Upper_Body_Clothing'].get(attr_data.get('Upper_Body_Clothing', 'Unknown'), 12), dtype=torch.long),
            'Lower_Body_Clothing': torch.tensor(self.attribute_maps['Lower_Body_Clothing'].get(attr_data.get('Lower_Body_Clothing', 'Unknown'), 9), dtype=torch.long),
            'Action': torch.tensor(self.attribute_maps['Action'].get(attr_data.get('Action', 'Unknown'), 0), dtype=torch.long),
        }
        
        # For accessories, create a binary indicator (has accessory or not)
        accessory_value = attr_data.get('Accessories', 'Unknown')
        has_accessory = 0 if accessory_value in ['Nothing', 'Unknown'] else 1
        labels['Accessories'] = torch.tensor(has_accessory, dtype=torch.float)
        
        return image, labels

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct_preds = {attr: 0 for attr in ['Gender', 'Age', 'Upper_Body_Clothing', 'Lower_Body_Clothing', 'Action']}
    total_samples = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        # Move data to device
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss for each attribute
        loss = 0
        for attr_name, output in outputs.items():
            if attr_name == 'Accessories':
                # Binary cross-entropy for binary classification
                attr_loss = torch.nn.BCEWithLogitsLoss()(output, labels[attr_name])
            else:
                # Cross-entropy for single-label classification
                attr_loss = criterion(output, labels[attr_name])
            
            loss += attr_loss
            
            # Calculate accuracy for single-label attributes
            if attr_name != 'Accessories':
                _, preds = torch.max(output, 1)
                correct_preds[attr_name] += (preds == labels[attr_name]).sum().item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    
    # Calculate epoch statistics
    epoch_loss = running_loss / total_samples
    epoch_acc = {attr: correct_preds[attr] / total_samples for attr in correct_preds}
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct_preds = {attr: 0 for attr in ['Gender', 'Age', 'Upper_Body_Clothing', 'Lower_Body_Clothing', 'Action']}
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            # Move data to device
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss for each attribute
            loss = 0
            for attr_name, output in outputs.items():
                if attr_name == 'Accessories':
                    # Binary cross-entropy for binary classification
                    attr_loss = torch.nn.BCEWithLogitsLoss()(output, labels[attr_name])
                else:
                    # Cross-entropy for single-label classification
                    attr_loss = criterion(output, labels[attr_name])
                
                loss += attr_loss
                
                # Calculate accuracy for single-label attributes
                if attr_name != 'Accessories':
                    _, preds = torch.max(output, 1)
                    correct_preds[attr_name] += (preds == labels[attr_name]).sum().item()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    # Calculate validation statistics
    val_loss = running_loss / total_samples
    val_acc = {attr: correct_preds[attr] / total_samples for attr in correct_preds}
    
    return val_loss, val_acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define paths
    data_dir = "attribute_data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_attr_file = os.path.join(data_dir, "train_attributes.json")
    val_attr_file = os.path.join(data_dir, "val_attributes.json")
    
    # Check if data exists
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Error: Training data not found. Run prepare_attribute_data.py first.")
        return
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AttributeDataset(train_dir, train_attr_file, transform=train_transform)
    val_dataset = AttributeDataset(val_dir, val_attr_file, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Define number of classes for each attribute based on P-DESTRE format
    num_classes_dict = {
        'Gender': 3,           # Male, Female, Unknown
        'Age': 9,              # 8 age groups + Unknown
        'Upper_Body_Clothing': 13, # 12 clothing types + Unknown
        'Lower_Body_Clothing': 10, # 9 clothing types + Unknown
        'Accessories': 1,      # Binary (has accessory or not)
        'Action': 14           # 13 action types + Unknown
    }
    
    # Create model
    model = AttributeClassifier(num_classes_dict)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training parameters
    num_epochs = 20
    best_val_loss = float('inf')
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = {attr: [] for attr in ['Gender', 'Age', 'Upper_Body_Clothing', 'Lower_Body_Clothing', 'Action']}
    val_accs = {attr: [] for attr in ['Gender', 'Age', 'Upper_Body_Clothing', 'Lower_Body_Clothing', 'Action']}
    
    # Create directory for saving models
    model_dir = os.path.join(data_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for attr in train_acc:
            print(f"{attr.capitalize()} - Train Acc: {train_acc[attr]:.4f}, Val Acc: {val_acc[attr]:.4f}")
        
        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for attr in train_acc:
            train_accs[attr].append(train_acc[attr])
            val_accs[attr].append(val_acc[attr])
        
        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_attribute_model.pth"))
            print("Saved best model checkpoint")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(model_dir, f"attribute_model_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_attribute_model.pth"))
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.legend()
    
    # Plot accuracy for each attribute
    for i, attr in enumerate(['Gender', 'Age', 'Upper_Body_Clothing', 'Lower_Body_Clothing', 'Action']):
        plt.subplot(2, 3, i+2)
        plt.plot(train_accs[attr], label='Train')
        plt.plot(val_accs[attr], label='Validation')
        plt.title(f'{attr.capitalize()} Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "training_curves.png"))
    plt.close()
    
    print("Training completed!")

if __name__ == "__main__":
    main() 