import cv2
import numpy as np

def count_fingers(contour):
    """Count fingers using convexity defects"""
    try:
        # Calculate convex hull for finger detection
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Get convexity defects
        defects = cv2.convexityDefects(contour, hull)
        
        # Count fingers using convexity defects
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate distance between points
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                # Calculate angle
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
                
                # If angle is less than 90 degrees, it's likely a finger
                if angle <= 90 and d > 10000:  # Lower threshold for better finger detection
                    finger_count += 1
        
        # Add 1 to finger count for the thumb (often not detected by convexity defects)
        if finger_count > 0:
            finger_count += 1
            
        return finger_count
    
    except Exception as e:
        print(f"Error counting fingers: {e}")
        return 0
