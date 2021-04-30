import cv2
import numpy as np


def filmulti(camid0: int, camid1: int):
    cap0 = cv2.VideoCapture(camid0)
    cap1 = cv2.VideoCapture(camid1)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            continue

        frame = np.hstack((frame0, frame1))
        cv2.imshow("multicam", frame)

        if cv2.waitKey(1) % 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filmulti(2, 4)
