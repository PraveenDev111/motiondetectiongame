import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import time
import random
from ultralytics import YOLO
from PIL import Image

# Import finger counting function
from count_fingers import count_fingers

# Import the gesture recognizer
try:
    from model_integration import GestureRecognizer
    # Initialize the gesture recognizer with the trained model
    gesture_recognizer = GestureRecognizer()
    USE_PYTORCH_MODEL = gesture_recognizer.model is not None
    if USE_PYTORCH_MODEL:
        print("PyTorch model loaded successfully for gesture recognition")
    else:
        print("PyTorch model not available, falling back to traditional CV methods")
except ImportError:
    print("model_integration.py not found, falling back to traditional CV methods")
    USE_PYTORCH_MODEL = False

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
        # Release the camera if it was previously opened
        if camera is not None:
            camera.release()
            camera = None
        
        # Simple camera initialization - this worked previously
        try:
            camera = cv2.VideoCapture(0)
            # Wait for camera to initialize
            time.sleep(1)
            
            if not camera.isOpened():
                print("Error: Could not open camera.")
                return None
                
            print("Camera initialized successfully.")
            return camera
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return None
    return camera

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
    
    # Step 1: Original frame
    processing_frames.append(("1. Original", frame.copy()))
    
    # Step 2: Convert to grayscale (explicitly required in assignment)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(gray_display, "GRAYSCALE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("2. Grayscale", gray_display))
    
    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blur_display = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    cv2.putText(blur_display, "BLUR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("3. Gaussian Blur", blur_display))
    
    # Step 4: Thresholding (explicitly required in assignment)
    # Use both binary and adaptive thresholding
    _, binary_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    binary_display = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2BGR)
    cv2.putText(binary_display, "BINARY THRESHOLD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("4. Binary Threshold", binary_display))
    
    # Step 5: Adaptive thresholding for better results in varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    adaptive_display = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
    cv2.putText(adaptive_display, "ADAPTIVE THRESHOLD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("5. Adaptive Threshold", adaptive_display))
    
    # Step 6: Skin detection in YCrCb color space for better hand isolation
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    skin_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(skin_display, "SKIN DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("6. Skin Detection", skin_display))
    
    # Step 7: Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    morph_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel)
    morph_display = cv2.cvtColor(morph_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(morph_display, "MORPHOLOGY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("7. Morphology", morph_display))
    
    # Step 8: Combine skin detection with thresholding for better hand isolation
    combined_mask = cv2.bitwise_and(morph_mask, adaptive_thresh)
    combined_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(combined_display, "COMBINED MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("8. Combined Mask", combined_display))
    
    # Step 9: Find and draw contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = frame.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.putText(contour_img, "CONTOURS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processing_frames.append(("9. Contours", contour_img))
    
    return frame, combined_mask, contours

def detect_gesture(frame):
    """Detect hand gesture using PyTorch model or traditional CV methods"""
    global processed_image, processing_frames

    try:
        # First try to use the PyTorch model if available
        if USE_PYTORCH_MODEL:
            try:
                # Use the trained PyTorch model for gesture recognition
                gesture, confidence = gesture_recognizer.recognize_gesture(frame)
                
                # Visualize the prediction
                viz_frame = gesture_recognizer.visualize_prediction(frame.copy(), gesture, confidence)
                processing_frames.append(("PyTorch Model Prediction", viz_frame))
                
                print(f"PyTorch model detected: {gesture} with confidence {confidence:.2f}")
                
                # If confidence is high enough, use the PyTorch prediction
                if confidence >= 0.7 and gesture != "unknown":
                    # Display detected gesture on the image
                    gesture_img = frame.copy()
                    cv2.putText(gesture_img, f"Detected: {gesture} (ML)", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    processing_frames.append(("Gesture Recognition (ML)", gesture_img))
                    return gesture
                    
                # If confidence is low, fall back to traditional methods
                print("PyTorch confidence too low, falling back to traditional methods")
            except Exception as e:
                print(f"Error using PyTorch model: {e}")
        
        # Preprocess the image
        original, thresh, contours = preprocess_image(frame)
        
        # Try using YOLO model for detection if available
        try:
            model = YOLO("yolov8n.pt")
            results = model(frame)
            result_image = results[0].plot()
            processing_frames.append(("YOLO Detection", result_image))
            processed_image = result_image
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            processed_image = original

        # Use traditional contour-based approach
        if contours and len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplified gesture recognition based on contour area and shape
            area = cv2.contourArea(largest_contour)

            if area < 1000:  # Too small, probably not a hand
                print(f"Contour area too small: {area}")
                return "unknown"

            # Calculate convex hull for contour
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)

            # Calculate solidity (ratio of contour area to hull area)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Calculate extent (ratio of contour area to bounding rectangle area)
            extent = float(area) / (w * h) if (w * h) > 0 else 0

            # Count fingers using convexity defects
            finger_count = count_fingers(largest_contour)

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
            cv2.putText(metrics_img, f"Extent: {extent:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(metrics_img, f"Finger Count: {finger_count}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processing_frames.append(("Metrics", metrics_img))

            print(f"Gesture metrics - Area: {area:.0f}, Aspect Ratio: {aspect_ratio:.2f}, Solidity: {solidity:.2f}, Extent: {extent:.2f}, Finger Count: {finger_count}")

            # Improved heuristics for gesture classification
            detected_gesture = "unknown"
            if finger_count <= 1 and solidity > 0.8:
                print("Detected ROCK gesture")
                detected_gesture = "rock"
            elif finger_count >= 4 and aspect_ratio > 0.5:
                print("Detected PAPER gesture")
                detected_gesture = "paper"
            elif finger_count == 2 and aspect_ratio < 0.8:
                print("Detected SCISSORS gesture")
                detected_gesture = "scissors"
            elif finger_count == 3 and aspect_ratio < 0.8 and solidity < 0.3:
                print("Detected LIZARD gesture")
                detected_gesture = "lizard"
            elif finger_count >= 3 and aspect_ratio < 0.8 and solidity < 0.3:
                print("Detected SPOCK gesture")
                detected_gesture = "spock"
            else:
                # Default to rock if we can't determine
                detected_gesture = "rock"

            # Display detected gesture on the image
            gesture_img = original.copy()
            cv2.putText(gesture_img, f"Detected: {detected_gesture} (CV)", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            processing_frames.append(("Gesture Recognition (CV)", gesture_img))

            return detected_gesture
        else:
            return "unknown"

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
    """Video streaming route. Put this in the src attribute of an img tag."""
    response = Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Set cache control headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/processed_feed')
def processed_feed():
    """Route for the processed image feed showing all steps"""
    response = Response(gen_processed_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Set cache control headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/capture')
def capture():
    """Capture and process the current frame to play the game"""
    global current_frame, game_result
    
    try:
        print("Capture endpoint called")
        
        # Always reset the current frame to ensure we're not using cached data
        current_frame = None
        
        # Try to get a new frame
        cam = initialize_camera()
        if cam is not None:
            # Read multiple frames to ensure we get the latest one
            for _ in range(3):  # Read a few frames to flush any buffered frames
                success, frame = cam.read()
                if success and frame is not None:
                    current_frame = frame.copy()
            
            if current_frame is not None:
                print("Successfully captured a new frame")
            else:
                print("Failed to capture a new frame")
        
        if current_frame is not None:
            # Save the frame to disk for debugging
            cv2.imwrite("debug_frame.jpg", current_frame)
            print(f"Frame saved for debugging, shape: {current_frame.shape}")
            
            # Detect user gesture with the fresh frame
            user_gesture = detect_gesture(current_frame)
            print(f"Detected gesture: {user_gesture}")
            
            # Ensure we have a valid user gesture
            if user_gesture == "unknown":
                user_gesture = "rock"  # Default to rock if unknown
                print("Defaulting to rock for unknown gesture")
            
            # Generate a new computer gesture each time
            computer_gesture = generate_computer_gesture()
            print(f"Computer gesture: {computer_gesture}")
            
            # Determine the winner
            result = determine_winner(user_gesture, computer_gesture)
            
            # Create a new game result (don't modify the global one directly)
            new_result = {
                "user_gesture": user_gesture,
                "computer_gesture": computer_gesture,
                "result": result,
                "timestamp": time.time()  # Add timestamp to prevent caching
            }
            
            # Update global game result
            game_result = new_result.copy()
            
            print(f"Game result: {new_result}")
            
            # Set cache control headers to prevent caching
            response = jsonify(new_result)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        else:
            print("No frame available for capture")
            # Create a default response with valid data
            default_response = {
                "user_gesture": "rock", 
                "computer_gesture": "paper", 
                "result": "Computer wins! Paper covers Rock",
                "timestamp": time.time()  # Add timestamp to prevent caching
            }
            
            # Set cache control headers
            response = jsonify(default_response)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
    except Exception as e:
        print(f"Error in capture endpoint: {str(e)}")
        # Return a default response with valid data
        default_response = {
            "user_gesture": "rock", 
            "computer_gesture": "scissors", 
            "result": "You win! Rock crushes Scissors",
            "timestamp": time.time()  # Add timestamp to prevent caching
        }
        
        # Set cache control headers
        response = jsonify(default_response)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

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
