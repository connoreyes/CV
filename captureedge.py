import cv2
import numpy as np

cap = cv2.VideoCapture(0) # start camera
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try 1 instead of 0.")

print("Press SPACE to capture. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Live Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        captured = frame.copy()
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

cap.release()
cv2.destroyAllWindows()

gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY).astype(np.float32)

sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

gx = cv2.filter2D(gray, -1, sobel_x)
gy = cv2.filter2D(gray, -1, sobel_y)

mag = np.sqrt(gx*gx + gy*gy)
mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)

cv2.imshow("Captured Image", captured)
cv2.imshow("Edges (Sobel magnitude)", mag)

cv2.imwrite("captured.jpg", captured)
cv2.imwrite("edges_sobel.jpg", mag)

cv2.waitKey(0)
cv2.destroyAllWindows()