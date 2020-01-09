import cv2

HAND_NUM = 2
HAND_FAIL_C_TH = 15

class trackingCV():
  def __init__(self, kcf = False):
    self.kcf = kcf
    self.cv_tracker_set()

  def cv_tracker_set(self):
    #self.tracker = cv2.TrackerKCF_create()
    #self.tracker = cv2.TrackerMIL_create()
    if self.kcf:
      self.tracker = cv2.TrackerKCF_create()
    else:
      self.tracker = cv2.TrackerMOSSE_create()

  def tracker_init(self, frame, left, top, width, height):
    self.cv_tracker_set()
    bbox = self.boxDiv(left, top, width, height)
    #bbox = cv2.selectROI(frame, False)
    ok = self.tracker.init(frame, bbox)

  def tracking(self, frame):
    track, bbox = self.tracker.update(frame)
    return track, bbox #tf, bbox
  
  def boxDiv(self, left, top, width, height, divP = 6):
    centerW = left + width / 2
    centerH = top + height / 2
    divW = int(width / divP)
    divH = int(height / divP)
    return (centerW - divW, centerH - divH, divW * 2, divH * 2)
    


class handTrackerCV():
  def __init__(self, kcf = False):
    self.ht = []
    self.htfailC = []
    self.htTrackTF = []
    self.kcf = kcf

    for _ in range(HAND_NUM):
      self.setHandDetector()
      self.htfailC.append(0)
      self.htTrackTF.append(False)
    self.left = 0
    self.right = 1


  def setHandDetector(self):
    self.ht.append(trackingCV(self.kcf))
  
  def setHandDetectFailCounterReset(self, i):
    self.htfailC[i] = 0
    self.htTrackTF[i] = True

  def setHandDetectFailCounter(self, i):
    self.htfailC[i] += 1
    if self.htfailC[i] > HAND_FAIL_C_TH:
      self.htTrackTF[i] = False
  
  def setTrackingBox(self, frame, left, top, width, height, headPosW = None):
    
    #print(frame.shape)
    h, w, c = frame.shape
  
    if headPosW is None:
      if left + width / 2 < w / 2:
        #right hand
        #print("right hand")
        self.setHandDetectFailCounterReset(self.right)
        self.ht[self.right].tracker_init(frame, left, top, width, height)
      else:
        #left hand
        #print("left hand")
        self.setHandDetectFailCounterReset(self.left)
        self.ht[self.left].tracker_init(frame, left, top, width, height)

    else:
      if left + width / 2 < headPosW:
        #right hand
        self.setHandDetectFailCounterReset(self.right)
        self.ht[self.right].tracker_init(frame, left, top, width, height)
      else:
        #left hand
        self.setHandDetectFailCounterReset(self.left)
        self.ht[self.left].tracker_init(frame, left, top, width, height)
    
  def trackings(self, frame):
    tfs = []
    bboxes = []
    fctfs = []
    for i, tracker in enumerate(self.ht):
      tf, box = tracker.tracking(frame)
      bboxes.append(box)
      tfs.append(tf)
      if self.htTrackTF[i]:
        fctfs.append(True)
      else:
        fctfs.append(False)
      self.setHandDetectFailCounter(i)
    return tfs, bboxes, fctfs


    


