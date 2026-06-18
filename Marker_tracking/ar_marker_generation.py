import cv2

# Define dictionary (6x6 grid, 250 IDs)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate marker ID 23 (size 300x300 pixels)
try:
    img = cv2.aruco.generateImageMarker(d, 23, 300)
except AttributeError:
    img = cv2.aruco.drawMarker(cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250), 23, 300)

# Save as JPG
cv2.imwrite("marker.jpg", img)
print("Saved as marker.jpg")