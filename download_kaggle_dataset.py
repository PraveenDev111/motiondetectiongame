import os
import zipfile
import subprocess
import shutil
import sys

def download_kaggle_dataset():
    """
    Download the Rock Paper Scissors dataset from Kaggle
    Dataset URL: https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset
    """
    print("Downloading Rock Paper Scissors dataset from Kaggle...")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    
    # Check if kaggle.json exists
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("\nKaggle API credentials not found.")
        print("Please follow these steps to set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section and click 'Create New API Token'")
        print("3. This will download a kaggle.json file")
        print(f"4. Place this file in {kaggle_dir} directory")
        print("5. Run this script again\n")
        
        # Create .kaggle directory if it doesn't exist
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
            print(f"Created directory: {kaggle_dir}")
        
        input("Press Enter to continue once you've set up your Kaggle API credentials...")
        
        if not os.path.exists(kaggle_json):
            print("Kaggle API credentials still not found. Exiting.")
            return False
    
    # Set permissions for kaggle.json
    os.chmod(kaggle_json, 0o600)
    
    # Download the dataset
    dataset_name = "sanikamal/rock-paper-scissors-dataset"
    output_dir = "kaggle_dataset"
    
    if os.path.exists(output_dir):
        print(f"Directory {output_dir} already exists. Removing...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    try:
        subprocess.check_call(["kaggle", "datasets", "download", dataset_name, "--path", output_dir, "--unzip"])
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def organize_dataset():
    """Organize the downloaded dataset into a format suitable for our model"""
    print("Organizing dataset...")
    
    # Define source and destination directories
    source_dir = "kaggle_dataset"
    dest_dir = "gesture_dataset"
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # The actual structure of the Kaggle dataset is:
    # kaggle_dataset/Rock-Paper-Scissors/train/rock
    # kaggle_dataset/Rock-Paper-Scissors/train/paper
    # kaggle_dataset/Rock-Paper-Scissors/train/scissors
    # kaggle_dataset/Rock-Paper-Scissors/test/rock
    # kaggle_dataset/Rock-Paper-Scissors/test/paper
    # kaggle_dataset/Rock-Paper-Scissors/test/scissors
    
    # Map dataset folders to our gesture names
    gesture_types = ["rock", "paper", "scissors"]
    data_splits = ["train", "test"]
    
    # Count total files copied for each gesture
    gesture_counts = {gesture: 0 for gesture in gesture_types}
    
    # Copy files from all splits (train, test) to our destination folders
    for gesture in gesture_types:
        dest_path = os.path.join(dest_dir, gesture)
        
        # Create destination folder if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        # Copy files from both train and test folders
        for split in data_splits:
            # Try both possible source paths (there are duplicated folders in the dataset)
            possible_src_paths = [
                os.path.join(source_dir, "Rock-Paper-Scissors", split, gesture),
                os.path.join(source_dir, "Rock-Paper-Scissors", "Rock-Paper-Scissors", split, gesture)
            ]
            
            for src_path in possible_src_paths:
                if os.path.exists(src_path):
                    file_count = 0
                    for filename in os.listdir(src_path):
                        if filename.endswith((".jpg", ".jpeg", ".png")):
                            src_file = os.path.join(src_path, filename)
                            # Add a prefix to avoid filename conflicts
                            dest_file = os.path.join(dest_path, f"{split}_{filename}")
                            shutil.copy2(src_file, dest_file)
                            file_count += 1
                    
                    if file_count > 0:
                        print(f"Copied {file_count} {gesture} images from {split} split")
                        gesture_counts[gesture] += file_count
    
    # Print summary of copied files
    print("\nDataset organization summary:")
    for gesture, count in gesture_counts.items():
        print(f"{gesture}: {count} images")
    
    # Create empty folders for lizard and spock gestures
    for folder in ["lizard", "spock"]:
        folder_path = os.path.join(dest_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created empty folder for {folder} gesture")
    
    print("Dataset organization complete!")
    return True

if __name__ == "__main__":
    if download_kaggle_dataset():
        organize_dataset()
        print("\nDataset is ready for training!")
        print("You can now run 'python gesture_model.py' to train the model.")
        print("Note: The dataset only contains rock, paper, and scissors gestures.")
        print("For lizard and spock gestures, you'll need to collect your own data.")
    else:
        print("Failed to download dataset. Please check your internet connection and Kaggle API credentials.")
