import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import time
import random
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Global variables
game_result = {"user_gesture": None, "computer_gesture": None, "result": None}
processing_frames = []
current_frame = None
processed_image = None
camera = None

def initialize_camera():
    """Initialize the camera and return it if successful"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        # Wait for camera to initialize
        time.sleep(1)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return None
    return camera

def generate_computer_gesture():
    """Generate a random gesture for the computer"""
    gestures = ["rock", "paper", "scissors"]
    return random.choice(gestures)

def determine_winner(user_gesture, computer_gesture):
    """Determine the winner based on the gestures"""
    if user_gesture == computer_gesture:
        return "Tie!"
    
    if user_gesture == "rock":
        return "You win!" if computer_gesture == "scissors" else "Computer wins!"
    elif user_gesture == "paper":
        return "You win!" if computer_gesture == "rock" else "Computer wins!"
    elif user_gesture == "scissors":
        return "You win!" if computer_gesture == "paper" else "Computer wins!"
    
    return "Invalid gesture"

def preprocess_image(frame):
    """Process the image through various steps for gesture recognition"""
    global processing_frames
    processing_frames = []  # Reset processing frames
    
    # Original frame
    processing_frames.append(("Original", frame.copy()))
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processing_frames.append(("Grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    processing_frames.append(("Gaussian Blur", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)))
    
    # Thresholding
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processing_frames.append(("Threshold", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the original image
    contour_img = frame.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    processing_frames.append(("Contours", contour_img))
    
    return frame, thresh, contours

def detect_gesture(frame):
    """Detect hand gesture using YOLO and OpenCV"""
    global processed_image, processing_frames
    
    try:
        # Preprocess the image
        original, thresh, contours = preprocess_image(frame)
        
        # Use YOLO model for detection if available
        detected_gesture = "unknown"
        try:
            model = YOLO("yolov8n.pt")
            results = model(frame)
            result_image = results[0].plot()
            processing_frames.append(("YOLO Detection", result_image))
            processed_image = result_image
            
            # In a real implementation, we would use YOLO to detect hand and classify gesture
            # For now we'll continue with our simplified approach
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            processed_image = original
        
        # For now, we'll use a simple algorithm based on contour area
        if contours and len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplified gesture recognition based on contour area and shape
            area = cv2.contourArea(largest_contour)
            
            if area < 1000:  # Too small, probably not a hand
                return "unknown"
                
            # Calculate convex hull for contour
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            # Calculate solidity (ratio of contour area to hull area)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Draw rectangle around the hand
            rect_img = original.copy()
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            processing_frames.append(("Hand Detection", rect_img))
            
            # Display the metrics on the image for debugging
            metrics_img = original.copy()
            cv2.putText(metrics_img, f"Area: {area:.0f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Aspect Ratio: {aspect_ratio:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Solidity: {solidity:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processing_frames.append(("Metrics", metrics_img))
            
            # Improved heuristics for gesture classification
            # Rock: compact shape (high solidity), aspect ratio close to 1
            # Paper: wide shape (higher aspect ratio), medium solidity
            # Scissors: elongated shape, lower solidity
            
            print(f"Gesture metrics - Area: {area:.0f}, Aspect Ratio: {aspect_ratio:.2f}, Solidity: {solidity:.2f}")
            
            if solidity > 0.85 and 0.8 < aspect_ratio < 1.3:
                detected_gesture = "rock"
            elif aspect_ratio > 1.3 and 0.65 < solidity < 0.85:
                detected_gesture = "paper"
            elif aspect_ratio < 0.8 and solidity < 0.7:
                detected_gesture = "scissors"
            else:
                # Default to rock if we can't determine
                detected_gesture = "rock"
                
            # Display detected gesture on the image
            gesture_img = original.copy()
            cv2.putText(gesture_img, f"Detected: {detected_gesture}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            processing_frames.append(("Gesture Recognition", gesture_img))
            
        return detected_gesture
            
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return "unknown"

def gen_frames():
    """Generate camera frames"""
    global current_frame
    
    cam = initialize_camera()
    if cam is None:
        return
    
    while True:
        try:
            success, frame = cam.read()
            if not success:
                print("Failed to read from camera")
                # Try to reinitialize camera
                cam = initialize_camera()
                if cam is None:
                    time.sleep(1)
                    continue
            else:
                current_frame = frame.copy()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in gen_frames: {str(e)}")
            time.sleep(0.1)

def gen_processed_frames():
    """Generate processed frames with visualization steps"""
    while True:
        if processing_frames and len(processing_frames) > 0:
            # Get dimensions from the first frame
            first_frame = processing_frames[0][1]
            if first_frame is not None:
                height, width, _ = first_frame.shape
                
                # Create a blank canvas to hold all processing steps
                combined_frame = np.zeros((height * len(processing_frames), width, 3), dtype=np.uint8)
                
                # Add each processing step to the combined frame
                for i, (name, frame) in enumerate(processing_frames):
                    if frame is not None and frame.shape == first_frame.shape:
                        # Add text label for this processing step
                        cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Add this frame to the combined image
                        combined_frame[i*height:(i+1)*height, 0:width] = frame
                
                ret, buffer = cv2.imencode('.jpg', combined_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Sleep to reduce CPU usage
        time.sleep(0.1)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the camera feed"""
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_feed')
def processed_feed():
    """Route for the processed image feed showing all steps"""
    return Response(gen_processed_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    """Capture and process the current frame to play the game"""
    global current_frame, game_result
    
    try:
        print("Capture endpoint called")
        
        if current_frame is None:
            print("No current frame available")
            # Try to get a new frame
            cam = initialize_camera()
            if cam is not None:
                success, frame = cam.read()
                if success:
                    current_frame = frame.copy()
                    print("Successfully captured a new frame")
                else:
                    print("Failed to capture a new frame")
            
        if current_frame is not None:
            # Save the frame to disk for debugging
            cv2.imwrite("debug_frame.jpg", current_frame)
            print(f"Frame saved for debugging, shape: {current_frame.shape}")
            
            # Detect user gesture
            user_gesture = detect_gesture(current_frame)
            print(f"Detected gesture: {user_gesture}")
            
            # Generate computer gesture
            computer_gesture = generate_computer_gesture()
            
            # Determine the winner
            result = determine_winner(user_gesture, computer_gesture)
            
            # Update game result
            game_result = {
                "user_gesture": user_gesture,
                "computer_gesture": computer_gesture,
                "result": result
            }
            
            return jsonify(game_result)
        else:
            print("No frame available for capture")
            return jsonify({"error": "No frame available", "user_gesture": "unknown", "computer_gesture": "rock", "result": "Error: No camera frame available"})
    except Exception as e:
        print(f"Error in capture endpoint: {str(e)}")
        # Return a default response to prevent the frontend from crashing
        return jsonify({"error": str(e), "user_gesture": "unknown", "computer_gesture": "rock", "result": f"Error: {str(e)}"})

@app.route('/get_result')
def get_result():
    """Get the current game result"""
    return jsonify(game_result)

if __name__ == '__main__':
    # Ensure templates and static folders exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Initialize camera
    initialize_camera()
    
    # Check if YOLO model exists, download if not
    if not os.path.exists("yolov8n.pt"):
        print("Downloading YOLO model...")
        os.system("pip install ultralytics")
        try:
            from ultralytics import YOLO
            YOLO("yolov8n.pt")
        except Exception as e:
            print(f"Error downloading YOLO model: {e}")
    
    # Run the app
    app.run(debug=True)
