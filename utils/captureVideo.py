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
        #print(frame)
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
  
  def release(self):
    self.video_capture.release()
    cv2.destroyAllWindows()
  
  def hsvExtraction(self, frame, left, top, w, h, bp = 4):
    l, r, t, b = self.boxTF(left, top, w, h)
    imgBox = frame[t:b, l:r]
    hc = imgBox.T[0].flatten().mean()
    s = imgBox.T[1].flatten().mean()
    v = imgBox.T[2].flatten().mean()
    #print(h, s, v)
    return hc, s, v
  
  def boxTF(self, left, top, w, h, bp = 2):
    #return left, right, top, bottm
    leftIdx = int(left + w / 2 - w/bp)
    if leftIdx < 0:
      leftIdx = 0

    topIdx = int(top + h/2 - h/bp) 
    if topIdx < 0:
      topIdx = 0

    rightIdx = int(left + w / 2 + w/bp)
    if rightIdx > self.WIDTH:
      rightIdx = self.WIDTH
    
    bottomIdx = int(top + h/2 + h/bp)
    if bottomIdx > self.HEIGHT:
      bottomIdx = self.HEIGHT
    
    return leftIdx, rightIdx, topIdx, bottomIdx
    


  def hsvSkinMasking(self, frame, left, top, width, height, hw = 15, sw = 90, vw = 90, dilate_iter = 4, bp = 1):
    #h, s, v = self.hsvExtraction(frame, left, top, width, height)
    #lower_skin = np.array((h - hw, s - sw, v - vw), dtype=np.uint8)
    #upper_skin = np.array((h + hw, s + sw, v + vw), dtype=np.uint8)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    l, r, t, b = self.boxTF(left, top, width, height, bp = bp)
   
    mframe = frame[t:b, l:r]

    #mframe = frame[int(top):int(top + height), int(left):int(left + width)]
    mask = cv2.inRange(mframe, lower_skin, upper_skin)
    mask = cv2.erode(mask, dilate_kernel, iterations=dilate_iter)

    mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iter) #dilate 膨張

    mask = cv2.GaussianBlur(mask, (5, 5), 100)

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