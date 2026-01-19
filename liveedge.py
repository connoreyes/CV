from edge_db import EdgeDatabase
import cv2
import numpy as np
def main():
    cap = cv2.VideoCapture(0) # capture frames 

    if not cap.isOpened(): # confirm camera is able to open
        raise RuntimeError("Error opening camera")

    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # vertical edge detection
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) # horizontal edge detection

    db = EdgeDatabase()
    while True:
        ret, frame = cap.read() # read every frame
        if not ret:
            continue
        
        # create exit key
        key = cv2.waitKey(1) & 0xFF
        # exit key logic
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit
        # convert to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # changed -1 to cv2.CV_16S in order to maintain signed values(negative gradients)
        gx = cv2.filter2D(gray, cv2.CV_16S, sobel_x).astype(np.int16) # convolution
        gy = cv2.filter2D(gray, cv2.CV_16S, sobel_y).astype(np.int16) # convolution
        # changed mag to np.int32 to prevent overflow and preserve correctness
        gx32 = gx.astype(np.int32)
        gy32 = gy.astype(np.int32)
        premag = (gx32**2 + gy32**2)
        mag = np.sqrt(premag) # combine edge strength
        mag = (mag / (mag.max() + 2e-6) * 255).astype(np.uint8) # scale mag

        # store in db
        if key == ord('s'):
            gx_blob = gx.tobytes() # store both gx and gy as BLOBS
            gy_blob = gy.tobytes()
            row_id = db.insert_gradients(gx, gy)
            print(f"Saved gradients to DB. id={row_id}, shape={gx.shape}, dtype={gx.dtype}")


        cv2.imshow("Live Feed", frame)
        cv2.imshow("Edges (Sobel magnitude)", mag)

    cap.release()
    cv2.destroyAllWindows()

main()