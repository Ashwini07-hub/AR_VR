import cv2
import cv2.aruco as aruco

# 1. Dictionary & directory
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0) # Camera started

while True:
    ret, frame = cap.read()
    
    # 2. Marker detecting
    corners, ids, rejected = detector.detectMarkers(frame)

    # 3. if marker found then draw
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"Marker found! ID: {ids}")

    cv2.imshow('AR Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
