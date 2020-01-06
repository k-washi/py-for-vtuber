import cv2
import numpy as np

dilate_kernel = np.ones((3,3), np.uint8)

class cpatureVideo():
  def __init__(self, WIDTH = 160, HEIGHT = 120, deviceID = 0):
    self.video_capture = cv2.VideoCapture(deviceID)
    self.frame = None
    self.HEIGHT = HEIGHT
    self.WIDTH = WIDTH

    
    #self.size_set()

  """
  def size_set(self):
    self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
    self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
  """
  def get_size(self):
    return self.WIDTH, self.HEIGHT

  def read(self, rgb = True):
    ret, frame = self.video_capture.read()

    if ret:
      if rgb:
        #BGR2RGB
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.resize(self.frame, self.WIDTH, self.HEIGHT)

      self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      return self.resize(self.frame, self.WIDTH, self.HEIGHT)
    
    return None
  
  def resize(self,frame, width, height):
    return cv2.resize(frame, dsize=(width, height))

  def rgb2gray(self, frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  
  def rgb2hsv(self, frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
  
  def hsvExtraction(self, frame, left, top, w, h, bp = 4):
    imgBox = frame[int(top + h/2 - h/bp) :int(top+h/2 + h/bp), int(left + w/2 - w/bp):int(left+w/2 + w/bp)]
    hc = imgBox.T[0].flatten().mean()
    s = imgBox.T[1].flatten().mean()
    v = imgBox.T[2].flatten().mean()
    #print(h, s, v)
    return hc, s, v

  def hsvMasking(self, frame, left, top, width, height, hw = 20, sw = 15, vw = 30, dilate_iter = 4, bp = 2):
    h, s, v = self.hsvExtraction(frame, left, top, width, height)
    lower_skin = np.array((h - hw, s - sw, v - vw), dtype=np.uint8)
    upper_skin = np.array((h + hw, s + sw, v + vw), dtype=np.uint8)

    mframe = frame[int(top + height/2 - height/bp) :int(top+height/2 + height/bp), 
                      int(left + width/2 - width/bp):int(left+width/2 + width/bp)]

    mask = cv2.inRange(mframe, lower_skin, upper_skin)

    mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iter) #dilate 膨張

    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    

    #findcontours 輪郭抽出
    #_,contours,hierarchy= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask
    
  
  def _show_raw(self):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    while True:
      rgb_frame = self.read()
      plt.imshow(rgb_frame)
      plt.pause(0.1)
      plt.cla()


if __name__ == "__main__":
  cv = cpatureVideo()
  cv._show_raw()