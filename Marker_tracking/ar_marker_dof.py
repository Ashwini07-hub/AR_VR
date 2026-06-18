import cv2
import cv2.aruco as aruco
import numpy as np

# 1. Setup: Dictionary and Detector
# Using DICT_4X4_50 as specified
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# 2. Camera Matrix: This is very important
# When building a real AR app, camera calibration is required
# For now, standard placeholder values are used
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1)) 

# 3. Marker size (assumed to be 5cm) and 3D points for the Cube
marker_size = 0.05
# 8 corners of the Cube (X, Y, Z)
# Negative Z means it will be drawn 'above' (upwards) the marker
cube_points = np.float32([
    [-0.025,  0.025, 0], [ 0.025,  0.025, 0], [ 0.025, -0.025, 0], [-0.025, -0.025, 0], # Base points
    [-0.025,  0.025, -0.05], [ 0.025,  0.025, -0.05], [ 0.025, -0.025, -0.05], [-0.025, -0.025, -0.05] # Top points
])

# Marker corners (Required for solvePnP)
obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# 4. Start Camera
cap = cv2.VideoCapture(0)

print("AR Cube starting... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Detect Marker
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            # Pose Estimation: Find the orientation and distance of the marker
            ret_pnp, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            
            if ret_pnp:
                # Project 3D points into 2D screen points
                imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                # Draw Cube:
                # 1. Base (Green)
                frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
                # 2. Pillars (Blue) - Lines connecting Base and Top
                for j in range(4):
                    frame = cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255, 0, 0), 2)
                # 3. Top (Red)
                frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

                # Display Marker ID
                cv2.putText(frame, f"ID: {ids[i][0]}", tuple(imgpts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Final AR Cube - Ashwini', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
