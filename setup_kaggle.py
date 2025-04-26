import os
import shutil
import sys

def setup_kaggle_credentials():
    """Set up Kaggle credentials by moving kaggle.json to the correct location"""
    print("Setting up Kaggle credentials...")
    
    # Source kaggle.json file in the current directory
    source_file = "kaggle.json"
    
    # Target directory in user's home folder
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    target_file = os.path.join(kaggle_dir, "kaggle.json")
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found in the current directory.")
        return False
    
    # Create .kaggle directory if it doesn't exist
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print(f"Created directory: {kaggle_dir}")
    
    # Copy the file
    try:
        shutil.copy2(source_file, target_file)
        print(f"Copied {source_file} to {target_file}")
        
        # Set permissions (important for Linux/Mac)
        try:
            os.chmod(target_file, 0o600)
            print("Set permissions on kaggle.json")
        except Exception as e:
            print(f"Note: Could not set file permissions (this is normal on Windows): {e}")
        
        print("Kaggle credentials set up successfully!")
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

if __name__ == "__main__":
    if setup_kaggle_credentials():
        print("\nNow you can run 'python gesture_model.py' to download the dataset and train the model.")
    else:
        print("\nFailed to set up Kaggle credentials. Please check the error messages above.")
