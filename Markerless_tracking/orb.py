import cv2
import numpy as np

# --- 1. Load Reference and Overlay Images ---
target_img = cv2.imread('tar.jpg')  
overlay_img = cv2.imread('ovr.jpeg') 

if target_img is None or overlay_img is None:
    print("Error: Could not load 'tar.png' or 'ovr.jpeg'. Please check files on your Desktop.")
    exit()

# Start Webcam with V4L2 stability configurations
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 2. Advanced ORB Setup (Increased features for stable locking) ---
orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
kp1, des1 = orb.detectAndCompute(target_img, None)

# Fast Flann-based matcher configuration for binary descriptors (ORB)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

h_target, w_target, _ = target_img.shape
overlay_img = cv2.resize(overlay_img, (w_target, h_target))

print("=== Highly Stable AR Tracking Engine Initialized ===")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a clean runtime copy to prevent visual feedback loop bleeding
    output_frame = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract frame keypoints
    kp2, des2 = orb.detectAndCompute(frame_gray, None)

    # Safe matching guard clause
    if des1 is not None and des2 is not None and len(kp2) > 30:
        # KNN Match to apply Lowe's Ratio Test for extreme stability
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Filter out bad/unstable feature matches
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # STRICT CONDITION: Require at least 35 persistent geometric matches to overlay
        if len(good_matches) > 35:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Robust Homography estimation using PROSAC/RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

            if H is not None and H.shape == (3, 3):
                h_frame, w_frame, _ = frame.shape
                
                # Warp perspective mapping
                warped_overlay = cv2.warpPerspective(overlay_img, H, (w_frame, h_frame))

                # Structural mask blending logic
                base_mask = np.ones((h_target, w_target), dtype=np.uint8) * 255
                warped_mask = cv2.warpPerspective(base_mask, H, (w_frame, h_frame))
                mask_inv = cv2.bitwise_not(warped_mask)
                
                # Cut a perfect hole and blend the target overlay
                frame_bg = cv2.bitwise_and(output_frame, output_frame, mask=mask_inv)
                output_frame = cv2.add(frame_bg, warped_overlay)

    # Output the clean stream
    cv2.imshow('Stable & Constant 6DoF Cube', output_frame)

    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
