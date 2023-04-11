import cv2
import numpy as np
import glob

frameSize = (500, 500)

out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, frameSize)

for filename in glob.glob('result/*.jpg'):
    img = cv2.imread(filename)
    out.write(img)

out.release()