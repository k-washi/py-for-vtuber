#wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#----------
import dlib
import cv2
import numpy as np
from imutils import face_utils

#-----------
from detector.tracking import trackingCV
from utils.kalmanFilter import Postionlkf


CV_MODEL_PATH = './model/haarcascade_frontalface_alt2.xml'
assert os.path.isfile(CV_MODEL_PATH ), 'haarcascade_frontalface_default.xml がない'
DLIB_MODEL_PATH = './model/shape_predictor_68_face_landmarks.dat'

#DLIB id: numpyに合わせて0から始まるものとする
DLIB_LEFT_EYE_ID = [42, 47]
DLIB_RIGHT_EYE_ID = [36, 41]
DLIB_NOSE_ID = 33
DLIB_CHIN_ID = 8
DLIB_REYE_ID = 36
DLIB_LEYE_ID = 45
DLIB_RM_ID = 48
DLIB_LM_ID = 54

#HWIDTH, HHEIGHT = 1280, 720
HWIDTH, HHEIGHT = 1920, 1080
LWIDTH, LHEIGHT = 320, 180

MAGNIW, MAGNIH = HWIDTH/float(LWIDTH), HHEIGHT/float(LHEIGHT)

dilate_kernel = np.ones((3,3), np.uint8)


#HEAD POSE ESTIMATION
FOCAL_LENGTH = HWIDTH
CENTER = (HWIDTH/2, HHEIGHT/2)
CAMERA_MATRIX = np.array([[FOCAL_LENGTH, 0, CENTER[0]],
                          [0, FOCAL_LENGTH, CENTER[1]],
                          [0, 0, 1]], dtype=np.float64)
 
#The Male mean interpupillary distance is 64.7 mm (https://en.wikipedia.org/wiki/Interpupillary_distance)
MODEL_POINT = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corne
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                      
                    ], dtype=np.float64)

DIST_COFFEIS = np.zeros((4,1)) #レンズ歪みなし


class faceDetectorDlib():
  def __init__(self):
    self.dlib_model_set()

  def dlib_model_set(self, file_path = DLIB_MODEL_PATH):
    self._detector = dlib.get_frontal_face_detector()
    #self._detector = cv2.CascadeClassifier(CV_MODEL_PATH)
    self._predictor = dlib.shape_predictor(file_path)

  def detect(self, frame):
    count = 0
    landmark = None
    low_frame = cv2.resize(frame, dsize=(LWIDTH, LHEIGHT))
    dets = self._detector(low_frame, 1) #rgb frame, upsample
    #dets = self._detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    face = None
    for det in dets:
      count += 1
      #print(det)
      #x,y,w,h = det[:]
      #face = dlib.rectangle(x, y, x+w, y+h)
      face = dlib.rectangle(int(det.left()*MAGNIW), int(det.top()*MAGNIH), 
                            int((det.left() + det.width())*MAGNIW), int((det.top() + det.height())*MAGNIH))
      #face = det
      landmark = self._predictor(frame, face)
      
      landmark = face_utils.shape_to_np(landmark)

      if count == 1:
        break
    
    return face, landmark
  
  def transfaceDetector(self, frame, width= HWIDTH, height = HHEIGHT, centerW = int(HWIDTH/2), centerH = int(HHEIGHT/2), dangle = 50, scale = 0.8):
    
    trans = cv2.getRotationMatrix2D((centerW, centerH), angle=dangle, scale=scale)
    frame = cv2.warpAffine(frame, trans, (width, height))
    det, landmark = self.detect(frame)

    if det is not None and landmark is not None:
      trans = np.array([trans[0,:], trans[1,:], [0,0,1]], dtype=np.float32)
      trans = np.linalg.inv(trans)
      det_a = np.array([[det.left()+det.width()/2], [det.top() + det.height()/2], [1]])
      det_a = np.dot(trans, det_a)[:2,0]

      det = dlib.rectangle(int(det_a[0] -det.width()/scale/2 ), int(det_a[1] - det.height()/scale/2), int(det_a[0] + det.width()/scale/2), int(det_a[1] + det.height()/scale/2))

      lp = np.array([[0], [0], [1]])
      
      for i in range(landmark.shape[0]):
        lp[0,0] = landmark[i, 0]
        lp[1,0] = landmark[i, 1]
        det_l = np.dot(trans, lp)[:2, 0]
        landmark[i, 0] = det_l[0]
        landmark[i, 1] = det_l[1]

      return det, landmark

    return None, None
  
  def eyeBlinkDetect(self, landmark):
    #(Real-Time Eye Blink Detection using Facial Landmarks)http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    #目をつぶったタイミングで目の位置がずれ、性能が低い。(眼鏡が原因??)
    left_eye = landmark[DLIB_LEFT_EYE_ID[0]: DLIB_LEFT_EYE_ID[1]+1]
    right_eye = landmark[DLIB_RIGHT_EYE_ID[0]: DLIB_RIGHT_EYE_ID[1]+1]

    a = np.linalg.norm(left_eye[1]-left_eye[5])
    b = np.linalg.norm(left_eye[2]-left_eye[4])
    c = np.linalg.norm(left_eye[0]-left_eye[3])
    left_eye_blinkP = (a + b) / (2 * c)

    a = np.linalg.norm(right_eye[1]-right_eye[5])
    b = np.linalg.norm(right_eye[2]-right_eye[4])
    c = np.linalg.norm(right_eye[0]-right_eye[3])
    right_eye_blinkP = (a + b) / (2 * c)    

    #print(left_eye_blinkP)
    #print(right_eye_blinkP)
  

  def eyeTracker(self, frame, landmark, left = True):
    eyeExpand = 10
    if left:
      eyeLandmark = landmark[DLIB_LEFT_EYE_ID[0]: DLIB_LEFT_EYE_ID[1]+1] 
    else:
      eyeLandmark = landmark[DLIB_RIGHT_EYE_ID[0]: DLIB_RIGHT_EYE_ID[1]+1]
    #print(eyeLandmark[:,0].min())
    eyew_min, eyew_max, eyeh_min, eyeh_max = eyeLandmark[:,0].min(), eyeLandmark[:,0].max(), eyeLandmark[:,1].min(), eyeLandmark[:,1].max()
    
    frame = frame[eyeh_min - eyeExpand: eyeh_max + eyeExpand, eyew_min - eyeExpand: eyew_max + eyeExpand]

    #lower_skin = np.array([40,80,70], dtype=np.uint8)
    #upper_skin = np.array([180,255,255], dtype=np.uint8)
    mframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, mframe = cv2.threshold(mframe, 44, 255, cv2.THRESH_BINARY)
    #mask = cv2.inRange(mframe, lower_skin, upper_skin)
    mask = cv2.erode(mframe, None, iterations=2)

    mask = cv2.dilate(mask, None, iterations=4) #dilate 膨張

    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #max area
    arearatio = None
    areacnt = None
    if len(contours) > 0:
      cnt = max(contours, key = lambda x: cv2.contourArea(x))   
      #print(cnt)

      epsilon = 0.001*cv2.arcLength(cnt,True)
      approx = cv2.approxPolyDP(cnt,epsilon,True)
      #print(approx)
      
      #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
      hull = cv2.convexHull(cnt)
      areahull = cv2.contourArea(hull) #凸を考慮した面積
      areacnt = cv2.contourArea(cnt) #凸を考慮しない面積
      arearatio=((areahull-areacnt)/areacnt)*100
      #print(len(contours))
    #print(arearatio)
    #print(areacnt)

    return cv2.bitwise_and(frame, frame, mask=mask)

    
    #print(eyeLandmark)
    #print(c)
    


  def faceDirection(self, landmark):
    #https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    image_points = np.array([landmark[DLIB_NOSE_ID],
                             landmark[DLIB_CHIN_ID],
                             landmark[DLIB_LEYE_ID],
                             landmark[DLIB_REYE_ID],
                             landmark[DLIB_LM_ID],
                             landmark[DLIB_RM_ID]
                             ], dtype=np.float64)
    #print(image_points)
    """
    [[ 987.  527.]
    [ 978.  709.]
    [1091.  444.]
    [ 899.  442.]
    [1046.  596.]
    [ 927.  600.]]
    """


    
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINT, image_points, CAMERA_MATRIX,DIST_COFFEIS, flags=cv2.SOLVEPNP_ITERATIVE)
    #print("SUCCESS: {0}".format(success))
    print("ROTATION: {0}".format(rotation_vector * 180 / np.pi))
    
    #print("TRANS: {0}".format(translation_vector))
    
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, CAMERA_MATRIX, DIST_COFFEIS)
    return rotation_vector, translation_vector, nose_end_point2D
  
  def mouthOpenDetector(self, landmark):
    pass


  def _show_landmark(self):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from utils.captureVideo import cpatureVideo
    import cv2

    fd = cpatureVideo(WIDTH=HWIDTH, HEIGHT=HHEIGHT, deviceID=0)
    tracker = trackingCV(kcf=True)
    facePos = Postionlkf(HWIDTH/2., HHEIGHT/2.)
    falseCounter = 0
    falseTh = 20
    pos = np.array([int(HWIDTH/2), int(HHEIGHT/2)]) 
    old_pos = np.array([int(HWIDTH/2), int(HHEIGHT/2)])

    #fig, ax = plt.subplots()

    while True:
      rgb_frame = fd.read()
      rgb_frame = rgb_frame.copy()
      #rgb_frame = fd.resize(rgb_frame, LWIDTH, LHEIGHT)
      
      if rgb_frame is None:
        continue

      gray_frame = fd.rgb2gray(rgb_frame)
      hsv_frame = fd.rgb2hsv(rgb_frame)
      det, landmark = self.detect(rgb_frame)
      
      if det is None:
        
        
        det, landmark = self.transfaceDetector(rgb_frame, dangle=50)
        if det is None:
          
          det, landmark = self.transfaceDetector(rgb_frame,dangle=-50)
        
      if det is not None and landmark is not None:
        rotV, tranV, noseP = self.faceDirection(landmark)
        cv2.line(rgb_frame, (int(landmark[DLIB_NOSE_ID][0]), int(landmark[DLIB_NOSE_ID][1])),
                  (int(noseP[0][0][0]), int(noseP[0][0][1])), (255, 0, 0), 2)
        #self.eyeBlinkDetect(landmark)
        #eyem = self.eyeTracker(rgb_frame,landmark)
        #rect = patches.Rectangle((det.left(), det.top()), det.width(), det.height(), fill=False)
        centerw = det.left() + det.width()/2
        centert = det.top() + det.height()/2
        width = det.width()/2
     
        #self.tracker_init(gray_frame, det.left(), det.top(), det.width(), det.height())
        tracker.tracker_init(hsv_frame, centerw - width, centert - width, width * 2, width * 2)

        #ax.add_patch(rect)
        '''
        left_eye = landmark[DLIB_LEFT_EYE_ID[0]: DLIB_LEFT_EYE_ID[1]+1]
        for k, (x, y) in enumerate(left_eye):
          plt.scatter(x, y)
          plt.text(x, y, k)
        
        '''
        #rgb_frame = eyem
        for k, (x,y) in enumerate(landmark):
          cv2.circle(rgb_frame, (int(x), int(y)), 3, (0,0,255), -1)
          #plt.scatter(x, y)
          #plt.text(x, y, k)
        
      
      track, rect = tracker.tracking(hsv_frame)
      
      
      if track:
        cv2.rectangle(rgb_frame, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 255, 0), 2)
        #rect = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False)
        #ax.add_patch(rect)
        
        falseCounter = 0
        pos, sigma = facePos.update(int(rect[0] + rect[2]/2), int(rect[1] + rect[3]/2) )
        old_pos = pos

        #print("pos {0}, Sigma {1}".format(pos, sigma))
        
        
      else:
        falseCounter += 1
        print("opos {0}".format(old_pos))
        if falseCounter < falseTh:
          pos, sigma = facePos.update(old_pos[0], old_pos[1])
        else:
          pos, sigma = facePos.update(int(HWIDTH/2), int(HHEIGHT/2) )
        #print("fail track")
      print("pos {0}".format(pos))
      cv2.circle(rgb_frame, (int(pos[0]), int(pos[1])), 10, (0,255,0), -1)
      cv2.imshow('img', rgb_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      #plt.imshow(rgb_frame)
      #plt.pause(0.01)
      #plt.cla()
      
    

if __name__ == "__main__":
  facedet = faceDetectorDlib()
  facedet._show_landmark()
  