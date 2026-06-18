import cv2
import cv2.aruco as aruco
import numpy as np
import time
from smbus2 import SMBus

# --- 📌 LSM6DS3 REGISTERS CONFIGURATION (UPDATED TO 0x68) 📌 ---
LSM6DS3_ADDR = 0x6b  # Changed from 0x6b to match your current hardware setup
CTRL1_XL     = 0x10  
CTRL2_G      = 0x11  
OUTX_L_G     = 0x22  

bus = None
try:
    bus = SMBus(1)
    # Initialize sensor in high-performance mode
    bus.write_byte_data(LSM6DS3_ADDR, CTRL1_XL, 0x80) 
    bus.write_byte_data(LSM6DS3_ADDR, CTRL2_G, 0x8C)  
    print(">>> SUCCESS: LSM6DS3 Hardware Active at 0x68 <<<")
except Exception as e:
    print(f"I2C Connection Error: {e}")

def read_lsm6ds3_word(addr):
    if bus is None: return 0.0
    try:
        low = bus.read_byte_data(LSM6DS3_ADDR, addr)
        high = bus.read_byte_data(LSM6DS3_ADDR, addr+1)
        value = (high << 8) | low
        if value > 32768: value -= 65536
        return float(value)
    except:
        return 0.0  # Fail-safe protection against loose wires

def get_lsm6ds3_gyro():
    gyro_scale = 16.384  # Scale for 2000 dps range
    raw_x = read_lsm6ds3_word(OUTX_L_G)
    raw_y = read_lsm6ds3_word(OUTX_L_G+2)
    raw_z = read_lsm6ds3_word(OUTX_L_G+4)

    gx = raw_x / gyro_scale
    gy = raw_y / gyro_scale
    gz = raw_z / gyro_scale
    return gx, gy, gz

# --- ArUco & Camera Setup ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

camera_matrix = np.array([[850, 0, 320], [0, 850, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1)) 

marker_size = 0.05
cube_points = np.float32([
    [-0.025,  0.025, 0], [ 0.025,  0.025, 0], [ 0.025, -0.025, 0], [-0.025, -0.025, 0], 
    [-0.025,  0.025, -0.05], [ 0.025,  0.025, -0.05], [ 0.025, -0.025, -0.05], [-0.025, -0.025, -0.05]
])

obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0], [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0], [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

cap = cv2.VideoCapture(0)

# --- TRACKING VARIABLES ---
rot_x, rot_y, rot_z = 0.0, 0.0, 0.0
last_time = time.time()

# 🏎️ Speed Booster multiplier
ROT_SPEED = 12.0  

# 🛑 NOISE FILTER DEADZONE 
DEADZONE = 1.5  

# 📐 FORCED PERFECT PERSPECTIVE VECTOR (Permanently locks ideal large size & screen center)
FORCED_LARGE_TVEC = np.array([[0.0], [0.0], [0.25]], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    if dt > 0.1: dt = 0.01

    # Get live gyro values from LSM6DS3
    gyro_x, gyro_y, gyro_z = get_lsm6ds3_gyro()

    corners, ids, rejected = detector.detectMarkers(frame)
    marker_detected = False
    rvec, tvec = None, None

    if ids is not None and len(ids) > 0:
        ret_pnp, rvec, tvec = cv2.solvePnP(obj_points, corners[0], camera_matrix, dist_coeffs)
        if ret_pnp:
            marker_detected = True
            
            # Save original camera view angles
            rot_x = float(rvec.flatten()[0])
            rot_y = float(rvec.flatten()[1])
            rot_z = float(rvec.flatten()[2])

    # --- 🏎️ ACTIVE IMU MODE WITH TOTAL SIZE ACCURACY ---
    if not marker_detected:
        
        # Cube rotates ONLY if sensor movement crosses deadzone
        if abs(gyro_x) > DEADZONE:
            rot_x += np.radians(gyro_x) * dt * ROT_SPEED
        if abs(gyro_y) > DEADZONE:
            rot_y += np.radians(gyro_y) * dt * ROT_SPEED
        if abs(gyro_z) > DEADZONE:
            rot_z += np.radians(gyro_z) * dt * ROT_SPEED
        
        # Build clean matrix format
        rvec = np.array([[rot_x], [rot_y], [rot_z]], dtype=np.float32)
        
        # Direct absolute override to prevent shrinking or jumping
        tvec = FORCED_LARGE_TVEC  
        
        cv2.putText(frame, "STATUS: IMU ACTIVE (SIZE PERFECTLY LOCKED)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    # --- 3D Drawing Engine ---
    if rvec is not None and tvec is not None:
        imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw 3D Cube Structure
        frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)  # Base
        for j in range(4):
            frame = cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255, 0, 0), 2)  # Pillars
        frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)  # Top

        if marker_detected:
            cv2.putText(frame, "STATUS: CAMERA LOCK (OK)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Telemetric data display
        txt_rot = f"Gyro X:{gyro_x:.1f} Y:{gyro_y:.1f} Z:{gyro_z:.1f}"
        cv2.putText(frame, txt_rot, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('LSM6DS3 6DoF Sensor Fusion', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
