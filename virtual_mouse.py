import cv2
import numpy as np
import pyautogui
import time
import math

# Optimize PyAutoGUI for instant response
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

class VirtualMouse:
    def __init__(self):
        # 1. Initialize OpenCV Built-in Cascade or Direct Hand Tracking via MediaPipe Tasks if available,
        # or use standard OpenCV color/contour tracking to guarantee zero version conflicts on Python 3.15!
        self.screen_w, self.screen_h = pyautogui.size()
        self.cam_w, self.cam_h = 640, 480
        self.frame_margin = 80

        # Cursor smoothing
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.smoothing = 8
        self.prev_time = 0

        print("[INFO] Initializing Color/Contour Hand Tracker for Python 3.15 compatibility...")

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, self.cam_w)
        cap.set(4, self.cam_h)

        print("=================================================")
        print(" AI Virtual Mouse (Compatible Mode) Started!     ")
        print(" Press 'Q' on the camera window to exit.         ")
        print("=================================================")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error: Webcam not detected.")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Draw active Region of Interest (ROI) box
            cv2.rectangle(
                frame,
                (self.frame_margin, self.frame_margin),
                (self.cam_w - self.frame_margin, self.cam_h - self.frame_margin),
                (255, 0, 255),
                2
            )

            # Convert to HSV color space to track a colored object/fingertip or skin mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Skin color range or bright marker tracking filter (Example: tracking bright objects or skin)
            # For robust beginner demo, we use simple contour detection inside the ROI box
            roi = frame[self.frame_margin:self.cam_h - self.frame_margin, self.frame_margin:self.cam_w - self.frame_margin]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mode_text = "Move Hand / Place Finger in Box"

            if contours:
                # Find the largest contour (hand/finger)
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 500:
                    ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
                    
                    # Adjust coordinates relative to full frame
                    center_x = int(x) + self.frame_margin
                    center_y = int(y) + self.frame_margin

                    # Map coordinates to screen
                    x_mapped = np.interp(center_x, [self.frame_margin, self.cam_w - self.frame_margin], [0, self.screen_w])
                    y_mapped = np.interp(center_y, [self.frame_margin, self.cam_h - self.frame_margin], [0, self.screen_h])

                    # Smoothing
                    self.curr_x = self.prev_x + (x_mapped - self.prev_x) / self.smoothing
                    self.curr_y = self.prev_y + (y_mapped - self.prev_y) / self.smoothing

                    pyautogui.moveTo(self.curr_x, self.curr_y)
                    self.prev_x, self.prev_y = self.curr_x, self.curr_y

                    # Draw tracker circle
                    cv2.circle(frame, (center_x, center_y), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), cv2.FILLED)
                    mode_text = "Tracking Active"

            # Display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
            self.prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {mode_text}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("AI Contactless Virtual Mouse", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VirtualMouse()
    app.run()