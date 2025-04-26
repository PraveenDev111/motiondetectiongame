import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import pickle
import shutil
import sys

# Define constants
GESTURES = ['rock', 'paper', 'scissors', 'lizard', 'spock']
NUM_CLASSES = len(GESTURES)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
MODEL_PATH = 'gesture_model.pth'
DATASET_PATH = 'gesture_dataset'  # Create this directory and add images

# Create dataset directory if it doesn't exist
os.makedirs(DATASET_PATH, exist_ok=True)

# Define image transformations for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define image transformations for validation/testing
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GestureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_model():
    """Create a pre-trained ResNet18 model with modified final layer for gesture classification"""
    # Load pre-trained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modify the final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    """Train the model and return training history"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {time_elapsed:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        print("-" * 60)
    
    return model, history

def save_model(model, path=MODEL_PATH):
    """Save the trained model"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path=MODEL_PATH):
    """Load a trained model"""
    model = create_model()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_for_prediction(frame):
    """Preprocess a frame for prediction with the trained model"""
    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the same transformations as during validation
    image_tensor = val_transforms(pil_image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_gesture(model, frame):
    """Predict the gesture in a frame using the trained model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Preprocess the frame
    image_tensor = preprocess_for_prediction(frame)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
    # Get the predicted class
    predicted_class = predicted.item()
    gesture = GESTURES[predicted_class]
    
    # Get confidence scores
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidence = probabilities[predicted_class].item()
    
    return gesture, confidence

def collect_training_data():
    """Function to collect training data from webcam"""
    print("Starting data collection mode...")
    
    # Create directories for each gesture if they don't exist
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    current_gesture = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display instructions
        instruction_frame = frame.copy()
        cv2.putText(instruction_frame, "Press keys for gestures:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(instruction_frame, "R: Rock, P: Paper, S: Scissors", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(instruction_frame, "L: Lizard, V: Spock", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(instruction_frame, "ESC: Exit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if current_gesture:
            cv2.putText(instruction_frame, f"Collecting: {current_gesture.upper()} ({frame_count})", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", instruction_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Set current gesture based on key press
        if key == ord('r'):
            current_gesture = 'rock'
            frame_count = 0
        elif key == ord('p'):
            current_gesture = 'paper'
            frame_count = 0
        elif key == ord('s'):
            current_gesture = 'scissors'
            frame_count = 0
        elif key == ord('l'):
            current_gesture = 'lizard'
            frame_count = 0
        elif key == ord('v'):
            current_gesture = 'spock'
            frame_count = 0
        elif key == 27:  # ESC key
            break
        
        # Save frame if a gesture is selected
        if current_gesture and frame_count < 200:  # Limit to 200 frames per gesture
            save_path = os.path.join(DATASET_PATH, current_gesture, 
                                    f"{current_gesture}_{frame_count}.jpg")
            cv2.imwrite(save_path, frame)
            frame_count += 1
            
            # Pause briefly to avoid duplicate frames
            time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")

def prepare_dataset():
    """Prepare the dataset for training"""
    image_paths = []
    labels = []
    
    # Collect all image paths and their corresponding labels
    for idx, gesture in enumerate(GESTURES):
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if os.path.exists(gesture_dir):
            for img_name in os.listdir(gesture_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(gesture_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(idx)
    
    # Split the dataset into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets
    train_dataset = GestureDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = GestureDataset(val_paths, val_labels, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader

def plot_training_history(history):
    """Plot the training and validation loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def check_kaggle_dataset():
    """Check if Kaggle dataset is available and properly organized"""
    # Check if the dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset directory {DATASET_PATH} not found.")
        return False
    
    # Check if we have data for at least rock, paper, scissors
    required_gestures = ['rock', 'paper', 'scissors']
    for gesture in required_gestures:
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if not os.path.exists(gesture_dir) or len(os.listdir(gesture_dir)) == 0:
            print(f"Missing data for {gesture} gesture.")
            return False
    
    # Count images for each gesture
    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if os.path.exists(gesture_dir):
            image_count = len([f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{gesture}: {image_count} images")
        else:
            print(f"{gesture}: 0 images")
    
    return True

def main():
    """Main function to run the training pipeline"""
    print("\nRock-Paper-Scissors-Lizard-Spock Gesture Recognition Model\n")
    
    # Check if Kaggle dataset is downloaded and organized
    if not check_kaggle_dataset():
        print("\nKaggle dataset not found or incomplete.")
        print("Would you like to:")
        print("1. Download the Kaggle Rock-Paper-Scissors dataset")
        print("2. Collect your own training data")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            try:
                # Try to import the download script
                from download_kaggle_dataset import download_kaggle_dataset, organize_dataset
                if download_kaggle_dataset():
                    organize_dataset()
                else:
                    print("Failed to download dataset. Exiting.")
                    return
            except ImportError:
                print("download_kaggle_dataset.py not found. Please run that script first.")
                return
        elif choice == '2':
            collect_training_data()
        else:
            print("Exiting.")
            return
    
    # Check if we have enough data to train
    has_data = any(os.path.exists(os.path.join(DATASET_PATH, gesture)) and 
                 len(os.listdir(os.path.join(DATASET_PATH, gesture))) > 0 
                 for gesture in ['rock', 'paper', 'scissors'])
    
    if not has_data:
        print("Error: Not enough training data available. Please collect data first.")
        return
    
    # Ask if user wants to train the model
    train_model_choice = input("\nDo you want to train the model now? (y/n): ").lower() == 'y'
    if not train_model_choice:
        print("Exiting without training.")
        return
    
    # Prepare the dataset
    train_loader, val_loader = prepare_dataset()
    
    # Create the model
    model = create_model()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Save the model
    save_model(trained_model)
    
    # Plot training history
    plot_training_history(history)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to {MODEL_PATH}")
    print("\nYou can now run app.py to use the trained model for gesture recognition.")

if __name__ == "__main__":
    main()
