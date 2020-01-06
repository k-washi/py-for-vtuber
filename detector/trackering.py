import cv2

class trackingCV():
  def __init__(self):
    pass

  def cv_tracker_set(self):
    #self.tracker = cv2.TrackerKCF_create()
    #self.tracker = cv2.TrackerMIL_create()
    self.tracker = cv2.TrackerMOSSE_create()

  def tracker_init(self, frame, left, top, width, height):
    self.cv_tracker_set()
    bbox = (left, top, width, height)
    #bbox = cv2.selectROI(frame, False)
    ok = self.tracker.init(frame, bbox)

  def tracking(self, frame):
    track, bbox = self.tracker.update(frame)
    return track, bbox #tf, bbox