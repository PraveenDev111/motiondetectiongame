import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import time
import random
from ultralytics import YOLO
from PIL import Image

# Import finger counting function
from count_fingers import count_fingers, visualize_fingers

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
# Default settings
active_detection_method = "pytorch" if USE_PYTORCH_MODEL else "opencv"
active_processing_method = "skin"
threshold_value = 127
blur_value = 5

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

def apply_edge_detection(frame):
    """Apply edge detection to the frame"""
    # Convert to grayscale if it's not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convert back to BGR for consistent display
    edge_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Add label (with background to prevent text overlap)
    cv2.rectangle(edge_frame, (5, 5), (150, 35), (0, 0, 0), -1)
    cv2.putText(edge_frame, "EDGE DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return edge_frame, edges

def preprocess_image(frame, method="skin"):
    """Process the image through various steps for gesture recognition"""
    global processing_frames, threshold_value, blur_value
    processing_frames = []  # Reset processing frames
    
    # Step 1: Original frame
    original = frame.copy()
    cv2.rectangle(original, (5, 5), (120, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(original, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("1. Original", original))
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(gray_display, (5, 5), (150, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(gray_display, "GRAYSCALE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("2. Grayscale", gray_display))
    
    # Step 3: Apply Gaussian blur to reduce noise
    # Use the blur value from slider
    blur_kernel = max(3, blur_value // 2 * 2 + 1)  # Ensure odd kernel size
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    blur_display = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(blur_display, (5, 5), (100, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(blur_display, "BLUR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("3. Gaussian Blur", blur_display))
    
    # Binary threshold
    _, binary_thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    binary_display = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(binary_display, (5, 5), (120, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(binary_display, "BINARY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("4. Binary Threshold", binary_display))
    
    # Edge Detection
    edge_display, edges = apply_edge_detection(frame)
    processing_frames.append(("5. Edge Detection", edge_display))
    
    # Canny edge detection
    canny_edges = cv2.Canny(blurred, threshold_value // 2, threshold_value)
    canny_display = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canny_display, (5, 5), (120, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(canny_display, "CANNY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("6. Canny Edge", canny_display))
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    adaptive_display = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(adaptive_display, (5, 5), (220, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(adaptive_display, "ADAPTIVE THRESHOLD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("7. Adaptive Threshold", adaptive_display))
    
    # Skin detection in YCrCb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    skin_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(skin_display, (5, 5), (200, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(skin_display, "SKIN DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("8. Skin Detection", skin_display))
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel)
    morph_display = cv2.cvtColor(morph_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(morph_display, (5, 5), (170, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(morph_display, "MORPHOLOGY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("9. Morphology", morph_display))
    
    # Choose the appropriate mask based on the method
    if method == "grayscale":
        final_mask = gray
    elif method == "binary" or method == "threshold":
        final_mask = binary_thresh
    elif method == "adaptive":
        final_mask = adaptive_thresh
    elif method == "edge":
        final_mask = edges
    elif method == "skin":
        final_mask = morph_mask
    elif method == "blur":
        final_mask = blurred
    elif method == "contour":
        # We'll create a contour mask in the next step
        final_mask = morph_mask  # Use the skin mask for contour detection
    elif method == "canny":
        final_mask = canny_edges
    else:  # Default to combined
        # Combine skin detection with thresholding
        final_mask = cv2.bitwise_and(morph_mask, adaptive_thresh)

    # Convert to BGR for display
    if len(final_mask.shape) == 2:
        final_display = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    else:
        final_display = final_mask.copy()
    
    cv2.rectangle(final_display, (5, 5), (200, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(final_display, "FINAL MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("10. Final Mask", final_display))
    
    # Find and draw contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = frame.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(contour_img, (5, 5), (150, 35), (0, 0, 0), -1)  # Background for text
    cv2.putText(contour_img, "CONTOURS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    processing_frames.append(("11. Contours", contour_img))
    
    return frame, final_mask, contours

def detect_gesture(frame, method="pytorch", processing="skin"):
    """Detect hand gesture using the specified method and processing technique"""
    global processed_image, processing_frames

    try:
        # Use PyTorch if requested and available
        if method == "pytorch" and USE_PYTORCH_MODEL:
            gesture, confidence = detect_gesture_pytorch(frame)
            if gesture != "unknown" and confidence >= 0.65:
                return gesture
                
        # Fall back to OpenCV methods if PyTorch fails or is not requested
        gesture, confidence = detect_gesture_opencv(frame, processing_method=processing)
        if gesture != "unknown" and confidence >= 0.6:
            return gesture
            
        # Try YOLO for object detection if gesture recognition failed
        try:
            model = YOLO("yolov8n.pt")
            results = model(frame)
            result_image = results[0].plot()
            processing_frames.append(("YOLO Detection", result_image))
            processed_image = result_image
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            
        return "unknown"
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return "unknown"

def detect_gesture_pytorch(frame):
    """Detect hand gesture using PyTorch model"""
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
            cv2.rectangle(gesture_img, (5, 180), (300, 210), (0, 0, 0), -1)  # Background for text
            cv2.putText(gesture_img, f"Detected: {gesture} (ML)", (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            processing_frames.append(("Gesture Recognition (ML)", gesture_img))
            return gesture, confidence
        
        return "unknown", confidence
    except Exception as e:
        print(f"Error using PyTorch model: {e}")
        return "unknown", 0.0

def detect_gesture_opencv(frame, processing_method="skin"):
    """Detect hand gesture using traditional OpenCV methods"""
    try:
        # Preprocess the image
        original, mask, contours = preprocess_image(frame, method=processing_method)
        
        # Traditional contour-based approach
        if contours and len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplified gesture recognition based on contour area and shape
            area = cv2.contourArea(largest_contour)

            if area < 1000:  # Too small, probably not a hand
                print(f"Contour area too small: {area}")
                return "unknown", 0.0

            # Calculate convex hull and defects for finger counting
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull)
            
            # Count extended fingers
            if defects is not None:
                finger_count = count_fingers(largest_contour, defects)
            else:
                finger_count = 0
                
            # Calculate geometric features
            hull_points = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull_points)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Calculate bounding rectangle features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = float(area) / (w * h) if (w * h) > 0 else 0
            
            # Calculate moments and circularity
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
                
            # Determine gesture based on features
            gesture = "unknown"
            confidence = 0.5  # Default confidence
                
            # Rock: closed fist (low finger count, high solidity)
            if finger_count <= 1 and solidity > 0.75:
                gesture = "rock"
                confidence = 0.7 + (solidity - 0.75) * 0.8  # Scale confidence
                
            # Paper: open hand (high finger count, medium solidity)
            elif finger_count >= 4 and aspect_ratio > 0.5 and aspect_ratio < 1.5:
                gesture = "paper"
                confidence = 0.7 + (finger_count / 5) * 0.3
                
            # Scissors: two fingers (finger count around 2-3, low solidity)
            elif finger_count in [2, 3] and solidity < 0.7:
                gesture = "scissors"
                confidence = 0.7 + (0.7 - solidity) * 0.5
                
            # Lizard: hand like puppet mouth (medium solidity, particular shape)
            elif finger_count in [1, 2] and solidity > 0.6 and solidity < 0.85:
                gesture = "lizard"
                confidence = 0.65
                
            # Spock: Vulcan salute (finger count 3-4, particular shape)
            elif finger_count in [3, 4] and solidity < 0.75:
                gesture = "spock"
                confidence = 0.65
                
            # Use visualization function to create informative result image
            viz_frame, _ = visualize_fingers(frame.copy(), largest_contour, defects)
            
            # Add gesture and confidence information to the visualization
            cv2.rectangle(viz_frame, (5, 380), (300, 470), (0, 0, 0), -1)  # Black background
            cv2.putText(viz_frame, f"Gesture: {gesture.upper()}", (10, 410), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Confidence: {confidence:.2f}", (10, 440), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Solidity: {solidity:.2f}", (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            processing_frames.append(("OpenCV Recognition", viz_frame))
            
            return gesture, confidence
            
        return "unknown", 0.0
    except Exception as e:
        print(f"Error in OpenCV detection: {e}")
        return "unknown", 0.0

def gen_frames():
    """Generate camera frames for the video feed"""
    cam = initialize_camera()
    if cam is None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('debug_frame.jpg', 'rb').read() + b'\r\n')
        return
        
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            global current_frame
            current_frame = frame.copy()
            
            # Add status text to the frame (with background to prevent overlap)
            status_method = "PyTorch AI" if active_detection_method == "pytorch" and USE_PYTORCH_MODEL else "OpenCV"
            cv2.rectangle(frame, (5, 5), (250, 35), (0, 0, 0), -1)  # Background for text
            cv2.putText(frame, f"Detection: {status_method}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def gen_processed_frames():
    """Generate processed frames for the image processing view"""
    global processing_frames, processed_image, current_frame, active_processing_method
    
    while True:
        # Process the current frame with the active method if we have a frame
        if current_frame is not None:
            try:
                # Process the current frame with the selected method
                _, final_mask, _ = preprocess_image(current_frame.copy(), method=active_processing_method)
                
                # Find the appropriate frame to display based on the active method
                method_to_index = {
                    "grayscale": 1,     # Grayscale frame
                    "blur": 2,          # Blur frame
                    "binary": 3,        # Binary threshold frame
                    "threshold": 3,     # Alias for binary threshold
                    "edge": 4,          # Edge detection frame
                    "canny": 5,         # Canny edge frame
                    "adaptive": 6,      # Adaptive threshold frame
                    "skin": 7,          # Skin detection frame
                    "contour": 10       # Contour frame
                }
                
                # Get index based on active method or default to final mask
                frame_index = method_to_index.get(active_processing_method, 9)
                
                # Ensure the index is valid
                if len(processing_frames) > 0 and frame_index < len(processing_frames):
                    # Get the frame for the active method
                    frame_title, selected_frame = processing_frames[frame_index]
                    
                    # Add a label to show what processing is active
                    label = active_processing_method.upper()
                    cv2.rectangle(selected_frame, (5, 40), (250, 70), (0, 0, 0), -1)
                    cv2.putText(selected_frame, f"Active: {label}", (10, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Fallback to showing the raw mask
                    if len(final_mask.shape) == 2:  # If grayscale
                        selected_frame = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
                    else:
                        selected_frame = final_mask.copy()
                    
                    # Add a label
                    label = active_processing_method.upper()
                    cv2.rectangle(selected_frame, (5, 5), (250, 35), (0, 0, 0), -1)
                    cv2.putText(selected_frame, f"Processing: {label}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', selected_frame)
                if not ret:
                    time.sleep(0.1)
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error in processed frames: {e}")
                time.sleep(0.1)
        else:
            # No frame available, show empty frame
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty_frame, "Camera not available", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', empty_frame)
            if not ret:
                time.sleep(0.1)
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add a small delay to reduce CPU usage
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
    global active_processing_method, active_detection_method, threshold_value, blur_value
    
    # Update parameters if provided in query string
    if 'processing' in request.args:
        processing = request.args.get('processing')
        if processing in ['edge', 'skin', 'threshold', 'contour', 'grayscale', 'binary', 'adaptive', 'blur', 'canny']:
            active_processing_method = processing
    
    if 'method' in request.args:
        method = request.args.get('method')
        if method in ['pytorch', 'opencv']:
            active_detection_method = method
    
    if 'threshold' in request.args:
        try:
            threshold_value = int(request.args.get('threshold'))
        except ValueError:
            pass
            
    if 'blur' in request.args:
        try:
            blur_value = int(request.args.get('blur'))
        except ValueError:
            pass
    
    response = Response(gen_processed_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Set cache control headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/set_detection_method', methods=['POST'])
def set_detection_method():
    """Set the active detection method"""
    global active_detection_method
    
    data = request.json
    method = data.get('method', 'pytorch')
    
    # Validate method
    if method not in ['pytorch', 'opencv']:
        return jsonify({'success': False, 'error': 'Invalid detection method'})
    
    # If PyTorch is requested but not available, return an error
    if method == 'pytorch' and not USE_PYTORCH_MODEL:
        return jsonify({'success': False, 'error': 'PyTorch model not available'})
    
    active_detection_method = method
    return jsonify({'success': True, 'method': active_detection_method})

@app.route('/set_processing_method', methods=['POST'])
def set_processing_method():
    """Set the active processing method"""
    global active_processing_method
    
    data = request.json
    method = data.get('method', 'skin')
    
    # Validate method
    if method not in ['edge', 'skin', 'threshold', 'contour', 'grayscale', 'binary', 'adaptive', 'blur', 'canny']:
        return jsonify({'success': False, 'error': 'Invalid processing method'})
    
    active_processing_method = method
    return jsonify({'success': True, 'method': active_processing_method})

@app.route('/set_processing_params', methods=['POST'])
def set_processing_params():
    """Set processing parameters like threshold and blur values"""
    global threshold_value, blur_value
    
    data = request.json
    threshold = data.get('threshold', 127)
    blur = data.get('blur', 5)
    
    # Store values in global variables
    threshold_value = int(threshold)
    blur_value = int(blur)
    
    return jsonify({
        'success': True, 
        'threshold': threshold_value,
        'blur': blur_value
    })

@app.route('/capture')
def capture():
    """Capture the current frame and process it for gesture recognition"""
    global current_frame, game_result, threshold_value, blur_value
    
    # If no frame is available, return error
    if current_frame is None:
        return jsonify({'error': 'No frame available'})
    
    # Get detection method and processing parameters from query parameters
    method = request.args.get('method', active_detection_method)
    processing = request.args.get('processing', active_processing_method)
    
    # Get threshold and blur values if provided
    if 'threshold' in request.args:
        threshold_value = int(request.args.get('threshold'))
    if 'blur' in request.args:
        blur_value = int(request.args.get('blur'))
    
    try:
        # Process frame
        frame_copy = current_frame.copy()
        
        # Detect user's gesture
        user_gesture = detect_gesture(frame_copy, method=method, processing=processing)
        print(f"Detected gesture: {user_gesture}")
        
        # Generate computer's gesture
        computer_gesture = generate_computer_gesture()
        print(f"Computer gesture: {computer_gesture}")
        
        # Determine winner
        result = determine_winner(user_gesture, computer_gesture)
        print(f"Result: {result}")
        
        # Store the result
        game_result = {
            "user_gesture": user_gesture,
            "computer_gesture": computer_gesture,
            "result": result
        }
        
        # Return the result
        return jsonify(game_result)
    except Exception as e:
        print(f"Error in capture: {e}")
        return jsonify({
            'error': str(e),
            'user_gesture': 'unknown',
            'computer_gesture': 'rock',
            'result': 'Error occurred. Please try again.'
        })

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
