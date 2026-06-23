import numpy as np
import cv2

# 1. Video Capture Setup
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 2. Shi-Tomasi Corner Detection Parameters
feature_params = dict(maxCorners=60,       # Reduced point limit to keep the screen clean
                       qualityLevel=0.3,
                       minDistance=20,     # Increased distance so points don't clutter together
                       blockSize=7)

# 3. Optimized Lucas-Kanade Parameters
lk_params = dict(winSize=(31, 31),
                  maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# 4. Capture Initial Frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Webcam stream not accessible.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

if p0 is not None:
    p0 = cv2.cornerSubPix(old_gray, p0, winSize=(5, 5), zeroZone=(-1, -1), criteria=subpix_criteria)

# Persistent mask for tracking trails
mask = np.zeros_like(old_frame)

print("=== Neat & Clean Sparse Tracking Active ===")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # SMART FIX: Slowly fade out older tracking lines so the screen stays clean
    # Multiplying by 0.85 reduces the brightness of old lines every frame
    mask = cv2.multiply(mask, 0.85)

    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            status = st.flatten() == 1
            good_new = p1[status]
            good_old = p0[status]

            # 5. Render Clean Trails and Points
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                
                # Draw the smooth moving line on the fading mask
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                
                # Draw a neat, high-contrast circle for the point
                cv2.circle(frame, (a, b), 4, (255, 0, 0), -1)          # Blue core
                cv2.circle(frame, (a, b), 5, (255, 255, 255), 1)      # White border for clarity

            # Composite the frame with the fading mask
            output_img = cv2.add(frame, mask)
            
            # Update frame states
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            output_img = frame.copy()
    else:
        output_img = frame.copy()

    # Dynamic Replenishment: Re-detect new points if count drops below 20
    if p0 is None or len(p0) < 20:
        tracking_mask = np.ones_like(frame_gray) * 255
        if p0 is not None:
            for pt in p0:
                cv2.circle(tracking_mask, tuple(pt.flatten().astype(int)), 25, 0, -1)
                
        new_corners = cv2.goodFeaturesToTrack(frame_gray, mask=tracking_mask, **feature_params)
        
        if new_corners is not None:
            new_corners = cv2.cornerSubPix(frame_gray, new_corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=subpix_criteria)
            if p0 is not None and len(p0) > 0:
                p0 = np.vstack((p0, new_corners))
            else:
                p0 = new_corners

    cv2.imshow('Neat & Smooth Lucas-Kanade Tracking', output_img)

    key = cv2.waitKey(20) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
