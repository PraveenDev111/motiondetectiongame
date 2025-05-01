import cv2
import numpy as np

def count_fingers(contour, defects):
    """Count fingers using convexity defects"""
    try:
        # Count fingers using convexity defects
        finger_count = 0
        
        # Define the threshold for angle and distance
        angle_thresh = 80.0  # Angle threshold for finger detection
        dist_thresh = 12000  # Distance threshold for finger detection
        
        # Get bounding box to determine hand size
        x, y, w, h = cv2.boundingRect(contour)
        hand_area = w * h
        
        # Adjust thresholds based on hand size
        if hand_area > 40000:
            dist_thresh = 20000
        elif hand_area < 15000:
            dist_thresh = 8000
        
        # Process defects to count fingers
        if defects is not None and len(defects) > 0:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate distance between points
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                # Calculate angle using the Law of Cosines
                # Handle potential math domain error (when denominator is close to zero)
                try:
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
                except:
                    angle = 90.0  # Default angle if calculation fails
                
                # If angle is less than threshold and distance is large enough, it's likely a finger
                if angle <= angle_thresh and d > dist_thresh:
                    finger_count += 1
        
        # Add 1 to finger count for the thumb (often not detected by convexity defects)
        # For better accuracy, don't blindly add 1
        if finger_count > 0:  # Only add thumb if we detected other fingers
            finger_count += 1
        
        # Cap the finger count at 5 (maximum possible on one hand)
        finger_count = min(finger_count, 5)
            
        return finger_count
    
    except Exception as e:
        print(f"Error counting fingers: {e}")
        return 0

def visualize_fingers(frame, contour, defects):
    """Visualize finger detection on the frame"""
    try:
        # Create a copy of the frame
        viz_frame = frame.copy()
        
        # Draw the contour of the hand
        cv2.drawContours(viz_frame, [contour], 0, (0, 255, 0), 2)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Define the threshold for angle
        angle_thresh = 80.0
        dist_thresh = 12000
        
        # Adjust thresholds based on hand size
        hand_area = w * h
        if hand_area > 40000:
            dist_thresh = 20000
        elif hand_area < 15000:
            dist_thresh = 8000
        
        # Draw finger tips and convexity defects
        finger_count = 0
        
        if defects is not None and len(defects) > 0:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate angle
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                try:
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
                except:
                    angle = 90.0
                
                # Draw finger tips (start, end points) and defect point (far)
                cv2.circle(viz_frame, start, 5, (0, 0, 255), -1)  # Start point - red
                cv2.circle(viz_frame, end, 5, (0, 255, 0), -1)    # End point - green
                
                # Only mark potential finger gaps
                if angle <= angle_thresh and d > dist_thresh:
                    cv2.circle(viz_frame, far, 5, (255, 0, 0), -1)  # Defect point - blue
                    finger_count += 1
        
        # Add text showing finger count
        finger_count = min(finger_count + 1, 5)  # Add 1 for thumb, cap at 5
        cv2.rectangle(viz_frame, (x, y - 30), (x + 150, y), (0, 0, 0), -1)
        cv2.putText(viz_frame, f"Fingers: {finger_count}", (x + 5, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return viz_frame, finger_count
        
    except Exception as e:
        print(f"Error visualizing fingers: {e}")
        return frame, 0
