import cv2
import cv2.aruco as aruco
import numpy as np

# 1. Setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Camera Matrix (Standard settings for testing)
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1)) 

# Real size of the marker (0.05 meters = 5cm)
marker_size = 0.05
obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            # Pose Estimation
            ret_pnp, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            
            if ret_pnp:
                # Calculate distance in CM
                distance = tvec[2][0] * 100 
                
                # --- Logic for Color & Status ---
                if distance > 40:
                    color = (0, 255, 0)      # Green: Safe
                    status = "SAFE"
                elif 20 <= distance <= 40:
                    color = (0, 255, 255)    # Yellow: Warning
                    status = "LOW SPEED"
                else:
                    color = (0, 0, 255)      # Red: Danger
                    status = "EMERGENCY STOP"

                # Draw Marker
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Display distance and status on screen
                info_text = f"ID: {ids[i][0]} | Dist: {distance:.1f}cm"
                cv2.putText(frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"STATUS: {status}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow('AR Safety Monitor - Ashwini', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
