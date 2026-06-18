import cv2
import cv2.aruco as aruco
import numpy as np

# 1. Setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Dummy Camera Matrix (Required for Pose estimation)
# In reality calibration is required, but this works fine for testing
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1)) 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            # Estimate Pose (Marker size is assumed to be 0.05 meters)
            # estimatePoseSingleMarkers is an older method, but it is simpler
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            
            # Draw Marker
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Draw 3D Axes (X-Red, Y-Green, Z-Blue)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

    cv2.imshow('AR Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
