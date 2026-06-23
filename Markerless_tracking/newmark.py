import cv2
import numpy as np

# --- Camera Parameters ---
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# --- Small and Constant 3D Cube Points ---
cube_points = np.float32([
    [-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0],       # Base
    [-20, -20, -40], [20, -20, -40], [20, 20, -40], [-20, 20, -40] # Top
])

# Tracker and Corner settings
lk_params = dict(winSize=(31, 31), maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
feature_params = dict(maxCorners=50, qualityLevel=0.05, minDistance=15, blockSize=5)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialization States
initialized = False
p_init = None
old_gray = None

print("=== Rigid Stable 6DoF Tracker Started ===")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not initialized:
        old_gray = frame_gray.copy()
        p_init = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        if p_init is not None and len(p_init) >= 8:
            p0 = p_init.copy()
            p_base_origin = p_init.copy()
            initialized = True
        continue

    if initialized and p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None and st is not None:
            # Flatten status array for boolean masking
            status = st.flatten() == 1
            
            if np.sum(status) >= 8:
                good_new = p1[status]
                good_base = p_base_origin[status]

                # Find Homography Matrix
                H, mask = cv2.findHomography(good_base, good_new, cv2.RANSAC, 5.0)
                
                if H is not None and H.shape == (3, 3):
                    # Decompose Homography Matrix
                    num, rs, ts, ns = cv2.decomposeHomographyMat(H, K)
                    
                    rvec, _ = cv2.Rodrigues(rs[0])
                    tvec = ts[0]

                    # Position and Size constraints
                    tvec[2] = 180  # Size constant locking
                    tvec[0] = 0
                    tvec[1] = 0

                    # 3D Cube Projection mapping
                    imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, K, dist_coeffs)
                    imgpts = np.int32(imgpts).reshape(-1, 2)

                    # --- Render Cube Line Structure ---
                    cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)  # Base (Green)
                    for i, j in zip(range(4), range(4, 8)):
                        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)  # Pillars (Blue)
                    cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)  # Top (Red)

                # Keep original tracking continuity
                p0 = good_new.reshape(-1, 1, 2)
                p_base_origin = good_base.reshape(-1, 1, 2)
                old_gray = frame_gray.copy()
            else:
                initialized = False
        else:
            initialized = False

    # Reset Pipeline if tracking points drop
    if not initialized or p0 is None or len(p0) < 8:
        p_init = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if p_init is not None and len(p_init) >= 8:
            p0 = p_init.copy()
            p_base_origin = p_init.copy()
            old_gray = frame_gray.copy()
            initialized = True

    cv2.imshow('Stable & Constant 6DoF Cube', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()