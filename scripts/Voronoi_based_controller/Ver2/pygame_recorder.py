import cv2
import pygame
import os
import glob
import time

class ScreenRecorder:
    """
        This class is used to record a PyGame surface and save it to a video file.
    """
    def __init__(self, width, height, fps, out_file='output.mp4'):
        """
        Initialize the recorder with parameters of the surface.
        :param width: Width of the surface to capture
        :param height: Height of the surface to capture
        :param fps: Frames per second
        :param out_file: Output file to save the recording
        """
        # define the codec and create a video writer object
        self.cnt = 0
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.timestr):
            os.makedirs(self.timestr)
        
    def capture_frame(self, surf):
        """
         Call this method every frame, pass in the pygame surface to capture.
        :param surf: pygame surface to capture
        :return: None
        """
        """
           
            Note: surface must have the dimensions specified in the constructor.
        """
        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(surf), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # write the frame
        cv2.imwrite(self.timestr + "/" + str(self.cnt) + ".jpg", pixels)
        self.cnt += 1

    def end_recording(self):
        """
        Call this method to stop recording.
        :return: None
        """
        # stop recording
# References
#   For more tutorials on cv2.VideoWriter, go to:
#   - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
#   - https://medium.com/@enriqueav/how-to-create-video-animations-using-python-and-opencv-881b18e41397
