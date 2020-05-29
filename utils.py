import scipy.io as sio
import numpy as np
from config import NUM_VIDEOS, K_FRAMES, IMG_WIDTH, IMG_HEIGHT, NUM_CLASS, IMG_PATH
from PIL import Image
import os

files_img_landmark = os.listdir(IMG_PATH)

def read_video_landmark():
    batch = np.zeros([NUM_VIDEOS, K_FRAMES + 1, IMG_HEIGHT, IMG_WIDTH, 3])
    batch_landmark = np.zeros([NUM_VIDEOS, K_FRAMES + 1, IMG_HEIGHT, IMG_WIDTH, 3])
    labels = np.zeros([NUM_VIDEOS], dtype=np.int32)
    files = np.random.choice(files_img_landmark, [NUM_VIDEOS])
    for i, file in enumerate(files):
        img_name = np.random.choice(os.listdir(IMG_PATH + file))
        img = np.array(Image.open(IMG_PATH + file + "/" + img_name))
        h, w = img.shape[0], img.shape[1]
        start0 = 0
        start1 = w // 2
        frames = np.zeros([K_FRAMES+1, IMG_HEIGHT, IMG_WIDTH, 3])
        landmarks = np.zeros([K_FRAMES+1, IMG_HEIGHT, IMG_WIDTH, 3])
        for j in range(K_FRAMES + 1):
            frame = np.array(Image.fromarray(np.uint8(img[:, start0 + j * 224:start0 + (j + 1) * 224])).resize([IMG_WIDTH, IMG_HEIGHT]))
            landmark = np.array(Image.fromarray(np.uint8(img[:, start1 + j * 224:start1 + (j + 1) * 224])).resize([IMG_WIDTH, IMG_HEIGHT]))
            frames[j] = frame
            landmarks[j] = landmark
        idx = np.array(range(K_FRAMES+1), dtype=np.int32)
        np.random.shuffle(idx)
        frames = frames[idx]
        landmarks = landmarks[idx]
        batch[i] = frames
        batch_landmark[i] = landmarks
        labels[i] = int(img_name[:-4])

    return batch/127.5-1.0, batch_landmark/127.5-1.0, labels





if __name__ == "__main__":
    read_video_landmark()