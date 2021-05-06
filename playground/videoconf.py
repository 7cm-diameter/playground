from os import mkdir
from os.path import exists, join

import cv2
from comprex.util import get_current_file_abspath


def film(camid: int, fps: int, width: int, height: int):
    data_dir = join(get_current_file_abspath(__file__), "data")
    if not exists(data_dir):
        mkdir(data_dir)

    cap = cv2.VideoCapture(camid)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    filename = join(data_dir, f"{fps}-{width}x{height}.MP4")
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("multicam", frame)
        video.write(frame)
        if cv2.waitKey(1) % 0xFF == ord("q"):
            break

    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    film(4, 30, 320, 240)
