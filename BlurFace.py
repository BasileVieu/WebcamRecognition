import cv2
import mediapipe as mp
import numpy as np
import time


def main():
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    p_time = 0

    while video_capture.isOpened() :
        success, frame = video_capture.read()

        frame = cv2.flip(frame, 1)

        if not success :
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f"FPS : {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Mediapipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
