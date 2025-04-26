import os
import cv2
import numpy as np
import torch
from gesture_model import load_model, predict_gesture, GESTURES

class GestureRecognizer:
    def __init__(self, model_path='gesture_model.pth'):
        """Initialize the gesture recognizer with a trained model"""
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = 0.7  # Minimum confidence to accept a prediction
        self.load_model()
        
    def load_model(self):
        """Load the trained PyTorch model if it exists"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        else:
            print(f"Model file not found at {self.model_path}")
            return False
    
    def recognize_gesture(self, frame):
        """Recognize gesture in the given frame"""
        if self.model is None:
            return "unknown", 0.0
        
        try:
            # Predict gesture using the model
            gesture, confidence = predict_gesture(self.model, frame)
            
            # Return the gesture if confidence is above threshold
            if confidence >= self.confidence_threshold:
                return gesture, confidence
            else:
                return "unknown", confidence
        except Exception as e:
            print(f"Error in gesture recognition: {str(e)}")
            return "unknown", 0.0
    
    def visualize_prediction(self, frame, gesture, confidence):
        """Visualize the prediction on the frame"""
        # Create a copy of the frame
        viz_frame = frame.copy()
        
        # Add text showing the predicted gesture and confidence
        cv2.putText(viz_frame, f"Gesture: {gesture.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz_frame, f"Confidence: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw a colored border based on confidence
        if confidence >= 0.9:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence >= 0.7:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw border
        border_thickness = 5
        h, w = viz_frame.shape[:2]
        viz_frame = cv2.rectangle(viz_frame, (0, 0), (w, h), color, border_thickness)
        
        return viz_frame

def test_camera_recognition():
    """Test the gesture recognition with webcam feed"""
    # Initialize the gesture recognizer
    recognizer = GestureRecognizer()
    
    # Check if model was loaded successfully
    if recognizer.model is None:
        print("Failed to load model. Please train the model first.")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Recognize gesture
        gesture, confidence = recognizer.recognize_gesture(frame)
        
        # Visualize prediction
        viz_frame = recognizer.visualize_prediction(frame, gesture, confidence)
        
        # Display the frame
        cv2.imshow("Gesture Recognition", viz_frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_recognition()
