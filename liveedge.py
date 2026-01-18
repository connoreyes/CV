import cv2
import numpy as np

cap = cv2.VideoCapture(0) # capture frames 

if not cap.isOpened(): # confirm camera is able to open
    raise RuntimeError("Error opening camera")

sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # vertical edge detection
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) # horizontal edge detection

while True:
    ret, frame = cap.read() # read every frame
    if not ret:
        continue
    
    cv2.imshow("Live Feed", frame)
    
    # create exit key
    key = cv2.waitKey(1) & 0xFF
    # exit key logic
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gx = cv2.filter2D(gray, -1, sobel_x) # convolution
    gy = cv2.filter2D(gray, -1, sobel_y) # convolution
    mag = np.sqrt((gx**2 + gy**2)) # combine edge strength
    mag = (mag / (mag.max() + 2e-6) * 255).astype(np.uint8) # scale mag

    cv2.imshow("Captured Image", frame)
    cv2.imshow("Edges (Sobel magnitude)", mag)

cap.release()
cv2.destroyAllWindows()

