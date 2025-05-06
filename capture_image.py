# capture_image.py
import cv2

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Press Q to Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('images/captured_image.jpg', frame)
            break

    cap.release()
    cv2.destroyAllWindows()

# Run this to test image capture
if __name__ == "__main__":
    capture_image()
