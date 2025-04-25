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
    try:
        # Close camera if it was previously opened but not working properly
        if camera is not None:
            camera.release()
            camera = None
            time.sleep(0.5)
        
        # Try to open the camera
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Wait for camera to initialize
        time.sleep(1.5)
        
        # Check if camera opened successfully
        if not camera.isOpened():
            print("Error: Could not open camera with DirectShow.")
            # Try fallback method
            camera = cv2.VideoCapture(0)
            time.sleep(1.5)
            
            if not camera.isOpened():
                print("Error: Could not open camera with fallback method.")
                return None
        
        # Read a test frame to ensure camera is working
        success, frame = camera.read()
        if not success or frame is None:
            print("Error: Camera opened but could not read frame.")
            camera.release()
            camera = None
            return None
            
        print("Camera initialized successfully.")
        return camera
    except Exception as e:
        print(f"Error initializing camera: {e}")
        if camera is not None:
            camera.release()
            camera = None
        return None

def generate_computer_gesture():
    """Generate a random gesture for the computer"""
    # Include Lizard and Spock in the possible gestures
    gestures = ["rock", "paper", "scissors", "lizard", "spock"]
    return random.choice(gestures)

def determine_winner(user_gesture, computer_gesture):
    """Determine the winner based on the gestures"""
    # If either gesture is unknown, computer wins
    if user_gesture == "unknown":
        return "Computer wins! (Gesture not recognized)"
    
    # If same gesture, it's a tie
    if user_gesture == computer_gesture:
        return "Tie!"
    
    # Define the winning relationships for Rock-Paper-Scissors-Lizard-Spock
    # Each key beats the gestures in its value list
    winning_rules = {
        "rock": ["scissors", "lizard"],       # Rock crushes Scissors, Rock crushes Lizard
        "paper": ["rock", "spock"],         # Paper covers Rock, Paper disproves Spock
        "scissors": ["paper", "lizard"],    # Scissors cuts Paper, Scissors decapitates Lizard
        "lizard": ["paper", "spock"],       # Lizard eats Paper, Lizard poisons Spock
        "spock": ["rock", "scissors"]       # Spock vaporizes Rock, Spock smashes Scissors
    }
    
    # Check if user's gesture beats computer's gesture
    if computer_gesture in winning_rules.get(user_gesture, []):
        # Get the reason for winning
        reason = get_winning_reason(user_gesture, computer_gesture)
        return f"You win! {reason}"
    else:
        # Get the reason for losing
        reason = get_winning_reason(computer_gesture, user_gesture)
        return f"Computer wins! {reason}"

def get_winning_reason(winner, loser):
    """Get the reason why one gesture beats another"""
    reasons = {
        ("rock", "scissors"): "Rock crushes Scissors",
        ("rock", "lizard"): "Rock crushes Lizard",
        ("paper", "rock"): "Paper covers Rock",
        ("paper", "spock"): "Paper disproves Spock",
        ("scissors", "paper"): "Scissors cuts Paper",
        ("scissors", "lizard"): "Scissors decapitates Lizard",
        ("lizard", "paper"): "Lizard eats Paper",
        ("lizard", "spock"): "Lizard poisons Spock",
        ("spock", "rock"): "Spock vaporizes Rock",
        ("spock", "scissors"): "Spock smashes Scissors"
    }
    
    return reasons.get((winner, loser), "")

def preprocess_image(frame):
    """Process the image through various steps for gesture recognition"""
    global processing_frames
    processing_frames = []  # Reset processing frames
    
    # Original frame
    processing_frames.append(("Original", frame.copy()))
    
    # Convert to YCrCb color space for better skin detection
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    processing_frames.append(("YCrCb Color Space", ycrcb))
    
    # Skin color detection in YCrCb space
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    processing_frames.append(("Skin Mask", cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    processing_frames.append(("Morphology", cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)))
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processing_frames.append(("Grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    processing_frames.append(("Gaussian Blur", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)))
    
    # Adaptive thresholding (works better across different lighting conditions)
    thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    processing_frames.append(("Adaptive Threshold", cv2.cvtColor(thresh_adaptive, cv2.COLOR_GRAY2BGR)))
    
    # Combine skin detection with adaptive threshold for better hand isolation
    combined_mask = cv2.bitwise_and(skin_mask, thresh_adaptive)
    processing_frames.append(("Combined Mask", cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)))
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the original image
    contour_img = frame.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    processing_frames.append(("Contours", contour_img))
    
    return frame, combined_mask, contours

def detect_gesture(frame):
    """Detect hand gesture using YOLO and OpenCV"""
    global processed_image, processing_frames
    
    try:
        # Preprocess the image
        original, mask, contours = preprocess_image(frame)
        
        # Use YOLO model for detection if available
        detected_gesture = "unknown"
        confidence_scores = {}
        yolo_result = None
        
        try:
            # Load YOLO model - we'll use it to detect the hand region first
            model = YOLO("yolov8n.pt")
            results = model(frame)
            result_image = results[0].plot()
            processing_frames.append(("YOLO Detection", result_image))
            yolo_result = results[0]
            processed_image = result_image
            
            # Check if YOLO detected any objects that could be a hand
            hand_detected = False
            for box in yolo_result.boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = float(box.conf[0])
                
                # YOLO's default model can detect people - we can use this to help locate hand regions
                if class_name == "person" and conf > 0.5:
                    hand_detected = True
                    print(f"Detected person with confidence: {conf:.2f}")
            
            if hand_detected:
                # If a person is detected, we have higher confidence in our hand detection
                confidence_scores["yolo_person"] = 0.7
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            processed_image = original
        
        # Proceed with contour analysis if contours were found
        if contours and len(contours) > 0:
            # Find the largest contour (likely to be the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < 1000:  # Too small, probably not a hand
                return "unknown"
            
            # Calculate convex hull and convexity defects
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = None
            try:
                defects = cv2.convexityDefects(largest_contour, hull)
            except Exception as e:
                print(f"Error calculating convexity defects: {e}")
            
            # Count fingers using convexity defects
            finger_count = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(largest_contour[s][0])
                    end = tuple(largest_contour[e][0])
                    far = tuple(largest_contour[f][0])
                    
                    # Calculate distance between points
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    
                    # Calculate angle
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
                    
                    # If angle is less than 90 degrees, it's likely a finger
                    if angle <= 90 and d > 30000:  # d is distance from contour to hull
                        finger_count += 1
            
            # Draw the defects on the image
            defects_img = original.copy()
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(largest_contour[s][0])
                    end = tuple(largest_contour[e][0])
                    far = tuple(largest_contour[f][0])
                    cv2.line(defects_img, start, end, [0, 255, 0], 2)
                    cv2.circle(defects_img, far, 5, [0, 0, 255], -1)
            
            processing_frames.append(("Convexity Defects", defects_img))
            
            # Calculate hull for area calculation
            hull_points = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull_points)
            
            # Calculate solidity (ratio of contour area to hull area)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Draw rectangle around the hand
            rect_img = original.copy()
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            processing_frames.append(("Hand Detection", rect_img))
            
            # Calculate additional metrics for better classification
            # Extent: ratio of contour area to bounding rectangle area
            extent = float(area) / (w * h) if (w * h) > 0 else 0
            
            # Calculate moments and centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Display the metrics on the image for debugging
            metrics_img = original.copy()
            cv2.putText(metrics_img, f"Area: {area:.0f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Aspect Ratio: {aspect_ratio:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Solidity: {solidity:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Extent: {extent:.2f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Finger Count: {finger_count}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processing_frames.append(("Metrics", metrics_img))
            
            # Log metrics for debugging
            print(f"Gesture metrics - Area: {area:.0f}, Aspect Ratio: {aspect_ratio:.2f}, "  
                  f"Solidity: {solidity:.2f}, Extent: {extent:.2f}, Finger Count: {finger_count}")
            
            # Calculate confidence scores for each gesture based on metrics
            confidence_scores["rock"] = 0.0
            confidence_scores["paper"] = 0.0
            confidence_scores["scissors"] = 0.0
            confidence_scores["lizard"] = 0.0
            confidence_scores["spock"] = 0.0
            
            # Rock: fist shape (high solidity, low finger count)
            if solidity > 0.80 and finger_count <= 1 and aspect_ratio < 1.5:
                confidence_scores["rock"] = 0.8
            
            # Paper: open palm (medium solidity, high finger count, wide aspect ratio)
            if 0.6 < solidity < 0.85 and finger_count >= 4 and aspect_ratio > 0.8:
                confidence_scores["paper"] = 0.8
            
            # Scissors: two fingers extended (medium solidity, finger count around 2)
            if 0.65 < solidity < 0.85 and finger_count == 2:
                confidence_scores["scissors"] = 0.8
            
            # Lizard: resembles puppet mouth (medium solidity, specific finger configuration)
            if 0.6 < solidity < 0.8 and finger_count == 2 and aspect_ratio < 0.8:
                confidence_scores["lizard"] = 0.7
            
            # Spock: Vulcan salute (lower solidity, specific finger configuration)
            if 0.5 < solidity < 0.7 and finger_count >= 3 and finger_count <= 4 and aspect_ratio < 1.0:
                confidence_scores["spock"] = 0.7
            
            # Find the gesture with highest confidence
            max_confidence = 0.0
            for gesture, confidence in confidence_scores.items():
                if gesture in ["rock", "paper", "scissors", "lizard", "spock"] and confidence > max_confidence:
                    max_confidence = confidence
                    detected_gesture = gesture
            
            # If no gesture has high enough confidence, return unknown
            if max_confidence < 0.5:
                detected_gesture = "unknown"
                
            # Display detected gesture on the image
            gesture_img = original.copy()
            cv2.putText(gesture_img, f"Detected: {detected_gesture} ({max_confidence:.2f})", (10, 180), 
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
