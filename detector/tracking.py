import cv2

HAND_NUM = 2

class trackingCV():
  def __init__(self):
    self.cv_tracker_set()

  def cv_tracker_set(self):
    self.tracker = cv2.TrackerKCF_create()
    #self.tracker = cv2.TrackerMIL_create()
    #self.tracker = cv2.TrackerMOSSE_create()

  def tracker_init(self, frame, left, top, width, height):
    self.cv_tracker_set()
    bbox = self.boxDiv(left, top, width, height)
    #bbox = cv2.selectROI(frame, False)
    ok = self.tracker.init(frame, bbox)

  def tracking(self, frame):
    track, bbox = self.tracker.update(frame)
    return track, bbox #tf, bbox
  
  def boxDiv(self, left, top, width, height, divP = 4):
    centerW = left + width / 2
    centerH = top + height / 2
    divW = width / divP
    divH = height / divP
    return (centerW - divW, centerH - divH, divW * 2, divH * 2)
    


class handTracker():
  def __init__(self):
    self.ht = []
    for _ in range(HAND_NUM):
      self.setHandDetector()

    self.left = 0
    self.right = 1


  def setHandDetector(self):
    self.ht.append(trackingCV())
  
  def setTrackingBox(self, frame, left, top, width, height, headPosW = None):
    
    #print(frame.shape)
    h, w, c = frame.shape
  
    if headPosW is None:
      if left + width / 2 < w / 2:
        #right hand
        print("right hand")
        self.ht[self.right].tracker_init(frame, left, top, width, height)
      else:
        #left hand
        print("left hand")
        self.ht[self.left].tracker_init(frame, left, top, width, height)

    else:
      if left + width / 2 < headPosW / 2:
        #right hand
        self.ht[self.right].tracker_init(frame, left, top, width, height)
      else:
        #left hand
        self.ht[self.left].tracker_init(frame, left, top, width, height)
    
  def trackings(self, frame):
    tfs = []
    bboxes = []
    for tracker in self.ht:
      tf, box = tracker.tracking(frame)
      tfs.append(tf)
      bboxes.append(box)
    
    return tfs, bboxes


    


