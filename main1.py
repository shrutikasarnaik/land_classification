import cv2
import numpy as np

img = cv2.imread("satellite3.png")
img = cv2.resize(img, (800, 800))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask_trees = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
mask_roads = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 50, 255]))
mask_buildings = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([20, 255, 255]))

mask_combined = cv2.bitwise_or(mask_trees, mask_roads)
mask_combined = cv2.bitwise_or(mask_combined, mask_buildings)
mask_free = cv2.bitwise_not(mask_combined)

overlay = np.zeros_like(img)

overlay[mask_trees == 255] = [0, 255, 0]
overlay[mask_roads == 255] = [128, 128, 128]
overlay[mask_buildings == 255] = [0, 0, 255]
overlay[mask_free == 255] = [0, 255, 255]

final = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

cv2.imshow("Land Classification", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
