import cv2


class cpatureVideo():
  def __init__(self, WIDTH = 160, HEIGHT = 120, deviceID = 0):
    self.video_capture = cv2.VideoCapture(deviceID)
    self.frame = None
    self.size_set(WIDTH, HEIGHT)

  def size_set(self, WIDTH, HEIGHT):
    self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

  def read(self, rgb = True):
    ret, frame = self.video_capture.read()

    if ret:
      if rgb:
        #BGR2RGB
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.frame

      self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      return self.frame
    
    return None
  
  def rgb2gray(self, frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  
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