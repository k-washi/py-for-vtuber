#wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#----------
import dlib
import cv2
from imutils import face_utils

#-----------
from detector.tracking import trackingCV

DLIB_MODEL_PATH = './model/shape_predictor_68_face_landmarks.dat'

class faceDetectorDlib():
  def __init__(self):
    self.dlib_model_set()

  def dlib_model_set(self, file_path = DLIB_MODEL_PATH):
    self._detector = dlib.get_frontal_face_detector()
    self._predictor = dlib.shape_predictor(file_path)
  
  def detect(self, frame):
    count = 0
    landmark = None
    dets = self._detector(frame, 1) #rgb frame, upsample
    face = None
    for det in dets:
      count += 1
      face = det
      landmark = self._predictor(frame, det)
      landmark = face_utils.shape_to_np(landmark)

      if count == 1:
        break
    
    return face, landmark
  
  
  def _show_landmark(self):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from utils.captureVideo import cpatureVideo
    import cv2

    fd = cpatureVideo(WIDTH=320, HEIGHT=160)
    tracker = trackingCV()

    fig, ax = plt.subplots()

    while True:
      rgb_frame = fd.read()
      gray_frame = fd.rgb2gray(rgb_frame)
      det, landmark = self.detect(gray_frame)
      if det is None:
        print("det is None")
      if landmark is None:
        print("landmark is None")
      if det is not None and landmark is not None:
        rect = patches.Rectangle((det.left(), det.top()), det.width(), det.height(), fill=False)
        centerw = det.left() + det.width()/2
        centert = det.top() + det.height()/2
        width = det.width()/3
     
        #self.tracker_init(gray_frame, det.left(), det.top(), det.width(), det.height())
        tracker.tracker_init(rgb_frame, centerw - width, centert - width, width * 2, width * 2)

        #ax.add_patch(rect)
        
        for k, (x,y) in enumerate(landmark):
          
          plt.scatter(x, y)
          plt.text(x, y, k)
      
      track, rect = tracker.tracking(rgb_frame)
      if True:
        if track:
          rect = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False)
          ax.add_patch(rect)
          print("suc track")
        else:
          print("fail track")

      plt.imshow(rgb_frame)
      plt.pause(0.01)
      plt.cla()
      
    

if __name__ == "__main__":
  facedet = faceDetectorDlib()
  facedet._show_landmark()
  