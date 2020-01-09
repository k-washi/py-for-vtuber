#---------
import numpy as np
import cv2
import logging
import time
from copy import deepcopy

#---------

from utils.captureVideo import cpatureVideo
from detector.tracking import trackingCV
from detector.faceDetector import faceDetectorVtuve, DLIB_NOSE_ID
from detector.handDetector import handDetectorVtube, handTrackerCV

formatter="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=formatter)

FaceTrackerVelAttenuationRate = 0.8
FaceOriginBackVelAttenuationRate = 0.3

HandTrackerVelAttenuationRate = 0.3
HandOriginBackVelAttenuationRate = 0.3


FPS_CALC_FRAMES = 30

RECORD_PATH = './output.mp4'
FRAME_RATE = 4

RECORD_FMT = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if __name__ == "__main__":
  from utils.config import configInit
  Config = configInit()

  capVideo = cpatureVideo(WIDTH=Config.CamWidth, HEIGHT=Config.CamHeight, deviceID=Config.CamID)

  ###############
  #-Face Init---#
  ###############

  #tracker for face
  faceTracker = trackingCV(kcf=True) #use opencv kcf tracker
  
  #detector for face
  faceDetector = faceDetectorVtuve()

  faceTrackingFailureCounter = 0
  faceLandmarkDetectFailureCounter = 0

  #recoder
  facePos = np.array([int(Config.CamWidth/2), int(Config.CamHeight/2)]) 
  faceTempPos = np.array([int(Config.CamWidth/2), int(Config.CamHeight/2)]) 
  faceTrackerPos = np.array([int(Config.CamWidth/2), int(Config.CamHeight/2)]) 
  faceTrackerVel = np.array([0, 0])

  facePoseRotInit = np.array([[0.], [0.], [0.]])   
  facePoseRot = np.array([[0.], [0.], [0.]])  
  faceDetectPoseRot = np.array([[0.], [0.], [0.]]) 

  faceNoseProdPos = None

  mouth_open_ratio = None

  ###############
  #-Hand Init---#
  ###############

  handDetector = handDetectorVtube()
  handTracker = handTrackerCV()

  #SSD MODEL LOAD
  handDetector.load_inference_graph()

  handPosInit = [np.array([int(Config.CamWidth/2 + Config.CamWidth/4), int(Config.CamHeight + Config.CamHeight/4)]), 
                  np.array([int(Config.CamWidth/2 - Config.CamWidth/4), int(Config.CamHeight + Config.CamHeight/4)])]
  handPos = deepcopy(handPosInit)
  handTempPos = deepcopy(handPosInit)
  handVel = [np.array([0, 0]), np.array([0, 0])]
  
  #######
  #-FPS-#
  #######
  fpsCounter = 0
  fpsTimer = time.time()
  nowFPS = 0
  
  
  RECORD_SIZE = (Config.CamWidth, Config.CamHeight)
  if not Config.PlotOK:
    #black Screen
    plotScreen = np.zeros((Config.CamHeight, Config.CamWidth, 3), np.uint8)
    

  else:
    plotNosePoint = None
    faceRect = None
    faceLandmark = None
    
    RECORD_SIZE = (int(Config.CamWidth/Config.PlotScaleDiv), int(Config.CamHeight/Config.PlotScaleDiv))
    if Config.Record:
      frameRecoder = cv2.VideoWriter(RECORD_PATH, RECORD_FMT, FRAME_RATE, RECORD_SIZE)

  fingerNumText = "finger no detect"
  handOpenText = "hand no detect"

    

  while True:
    frame = capVideo.read() #get rgb frame
    
    ###############
    #-Face Handle-#
    ###############

    faceRect, faceLandmark = faceDetector.faceDetect(frame)

    if faceRect is not None or faceLandmark is not None:
      faceLandmarkDetectFailureCounter = 0

      faceDetectPoseRot, _, plotNosePoint = faceDetector.faceDirection(faceLandmark)
    
      faceTracker.tracker_init(frame, faceRect.left(), faceRect.top(), faceRect.width(), faceRect.height())
      facetrackingTF, faceRect = faceTracker.tracking(frame)
      if facetrackingTF:
        faceTrackerPos[0], faceTrackerPos[1] = int(faceRect[0] + faceRect[2]/2), int(faceRect[1] + faceRect[3]/2)
      
      mouth_open_ratio = faceDetector.mouthOpenDetector(faceLandmark)

      faceTempPos = faceDetector.getFaceCenterLandmark(faceLandmark)


    else:
      faceLandmarkDetectFailureCounter += 1

      facetrackingTF, faceRect = faceTracker.tracking(frame)
      if facetrackingTF:
        faceTrackingFailureCounter = 0

        faceRectW = faceRect[0] + faceRect[2]/2
        faceRectH = faceRect[1] + faceRect[3]/2

        faceTrackerVel[0], faceTrackerVel[1] = int(faceRectW - faceTrackerPos[0]), int(faceRectH - faceTrackerPos[1])
        faceTrackerPos[0], faceTrackerPos[1] = faceRectW, faceRectH

        faceTempPos[0], faceTempPos[1] = faceTempPos[0] + faceTrackerVel[0], faceTempPos[1] + faceTrackerVel[1]
      
      else:
        faceTrackingFailureCounter += 1
        if faceTrackingFailureCounter < Config.TrakingFailureTh:
          faceTrackerVel = faceTrackerVel * FaceTrackerVelAttenuationRate
          faceTempPos[0], faceTempPos[1] = faceTempPos[0] + faceTrackerVel[0], faceTempPos[1] + faceTrackerVel[1]

        else:
          originBackVelW = int((Config.CamWidth/2 - faceTempPos[0]) * FaceOriginBackVelAttenuationRate)
          originBackVelH = int((Config.CamHeight/2 - faceTempPos[1]) * FaceOriginBackVelAttenuationRate) 
          faceTempPos[0], faceTempPos[1] = originBackVelW, originBackVelH

    if np.linalg.norm(facePos - faceTempPos) > Config.MoveOffset:
      facePos = deepcopy(faceTempPos)
      facePoseRot = deepcopy(faceDetectPoseRot)
      
      facePoseRot = faceDetector.faceDirRange(facePoseRot, facePoseRotInit, x=Config.RotXRange, y=Config.RotYRange, z=Config.RotZRange)


      if plotNosePoint is not None:
        faceNoseProdPos = deepcopy(plotNosePoint)
        
    if faceLandmarkDetectFailureCounter > Config.FaceLandmarkFailureTh:
      #TODO facePoseRotInitに初期値を与えるようにする。
      #制限
      facePoseRot = deepcopy(facePoseRotInit)
    
    
    ###############
    #-Hand Handle-#
    ###############
    #TODO 手のトラッキングが残り続けるため、しきい値以上認識しなかった場合、消す処理を挟む必要がある
    handBoxes, handScores = handDetector.detect_objects(frame)
    handBoxes, handScores = handDetector.score_Classifier(handBoxes, handScores, Config.CamWidth, Config.CamHeight)
    handCounter = 0
    handOverLap = False
    handTempBox = (0, 0, 0, 0)
    handOpenRatios, fingerNums = [0., 0.], [0, 0]
    
    #Overlapしていない最大2つの手を認識し、トラッキング用のバウンディングボックスを設定する。
    
    for handBox in handBoxes:
      handCounter += 1
      if handCounter == 1:
        handTempBox = handBox
      else:
        handBoxCenterW = handBox[0] + handBox[2] / 2.
        handBoxCenterH = handBox[1] + handBox[3] / 2. 
        if handTempBox[0] < handBoxCenterW and handBoxCenterW < handTempBox[0] + handTempBox[2] and handTempBox[1] < handBoxCenterH and handBoxCenterH < handTempBox[1] + handTempBox[3]:
          handOverLap = True
      
      if not handOverLap:
        handTracker.setTrackingBox(frame, handBox[0], handBox[1], handBox[2], handBox[3], headPosW=facePos[0])
      
      if handCounter >= 2:
        break
    
    #handTracker.left => 0
    #handTracker.right => 1
    handTrackingTFs, handRects, handTrackingFailThTF = handTracker.trackings(frame)
    print("HANDTH")
    print(handTrackingTFs, handRects, handTrackingFailThTF)
    for i, handTrackingTF in enumerate(handTrackingTFs):
      if handTrackingTF:
        handRect = handRects[i]
        
        handRPosW, handRPosH = int(handRect[0] + handRect[2]/2), int(handRect[1] + handRect[3]/2)
        if handTrackingFailThTF[i]:
          handVel[i][0], handVel[i][1] = int(handRPosW - handTempPos[i][0]), int(handRPosH - handTempPos[i][1])
          print("hand {0}".format(handVel))
          #handTempPos[i][0], handTempPos[i][1] = handRPosW, handRPosH
        #handOpen and finger num counter
        hsv_frame = capVideo.rgb2hsv(frame)
        handMask = capVideo.hsvSkinMasking(hsv_frame, handRect[0], handRect[1], handRect[2], handRect[3])
        handOpenRatio, fingerNum = handDetector.handOpenDtector(handMask)
        handOpenRatios[i] = handOpenRatio
        fingerNums[i] = fingerNum 
      
        handTracker.setHandDetectFailCounter(i)

      if handTrackingFailThTF[i]:
      
        if handTrackingTF:
          handTempPos[i][0], handTempPos[i][1] = int(handTempPos[i][0] + handVel[i][0]), int(handTempPos[i][1] + handVel[i][1])
        else:
          
          handVel[i] = handVel[i] * HandTrackerVelAttenuationRate 
          handTempPos[i][0], handTempPos[i][1] = int(handTempPos[i][0] + handVel[i][0]), int(handTempPos[i][1] + handVel[i][1]) 
      else:
        
        handVel[i][0], handVel[i][1] = int(handPosInit[i][0] - handTempPos[i][0]), int(handPosInit[i][1] - handTempPos[i][1])
        handTempPos[i][0], handTempPos[i][1] = handTempPos[i][0] + handVel[i][0] * HandOriginBackVelAttenuationRate, handTempPos[i][1] + handVel[i][1]*HandOriginBackVelAttenuationRate
    
      if np.linalg.norm(handPos[i] - handTempPos[i]) > Config.MoveOffset:
        handPos = deepcopy(handTempPos)
    #print(handTrackingFailThTF)
    print(handPos, handTempPos, handVel, handPosInit)
        

    #---FPS---

    fpsCounter += 1
    if fpsCounter > FPS_CALC_FRAMES:
      nowFPS = int(FPS_CALC_FRAMES/(time.time() - fpsTimer))
      fpsTimer = time.time()
      fpsCounter = 0
      

    #---Plot---

    if Config.PlotOK:
      plotScreen = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      if Config.PlotBinarization:
        imBool = cv2.cvtColor(plotScreen, cv2.COLOR_RGB2GRAY) > 128
        r,g,b = 255, 128, 32
        plotScreen[:,:,0] = imBool * r
        plotScreen[:,:,1] = imBool * g
        plotScreen[:,:,2] = imBool * b
      if faceLandmarkDetectFailureCounter == 0:
        for k, (x,y) in enumerate(faceLandmark):
          cv2.circle(plotScreen, (int(x), int(y)), 3, (0,0,255), -1)
        if faceNoseProdPos is not None:
          cv2.line(plotScreen, (int(faceLandmark[DLIB_NOSE_ID][0]), int(faceLandmark[DLIB_NOSE_ID][1])),
                  (int(faceNoseProdPos[0][0][0]), int(faceNoseProdPos[0][0][1])), (255, 0, 0), 5)
          
      if facetrackingTF:
        cv2.rectangle(plotScreen, (int(faceRect[0]), int(faceRect[1])), 
                                  (int(faceRect[0] + faceRect[2]), int(faceRect[1] + faceRect[3])), (0, 255, 0))
      
      cv2.circle(plotScreen, (facePos[0], facePos[1]), 8, (0, 255, 0), 2)
    
    #FPS
    fpsText = "FPS: {:.4f}".format(nowFPS)
    cv2.putText(plotScreen, fpsText, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
     

    #face data plot
    #mouth
    if mouth_open_ratio is not None:
      mouthText = "Mouth Open Ratio: {:.4f}".format(mouth_open_ratio)
      cv2.putText(plotScreen, mouthText, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    
    if facePoseRot is not None:
      textRot = facePoseRot * 180. / np.pi
      rotText = "Face Rotation: x {:.4f}, y {:.4f}, z {:.4f}".format(textRot[0][0], textRot[1][0], textRot[2][0])
      cv2.putText(plotScreen, rotText, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    
    
    textHShift = 0
    for i, handRect in enumerate(handRects):
      if handTrackingTFs[i]:
        cv2.rectangle(plotScreen, (int(handRect[0]), int(handRect[1])), 
                                    (int(handRect[0] + handRect[2]), int(handRect[1] + handRect[3])), (0, 255, 0),5)
        handCenterW = int(handRect[0] + handRect[2]/2)
        if handCenterW < facePos[0]:
          if handOpenRatio is not None:
            handOpenText = "Right Hand Open Ratio: {:.4f}".format(handOpenRatios[i])
            fingerNumText = "Right Finger Num: {0}".format(fingerNums[i])
        else:
          if handOpenRatio is not None:
            handOpenText = "Left Hand Open Ratio: {:.4f}".format(handOpenRatios[i])
            fingerNumText = "Left Finger Num: {0}".format(fingerNums[i])
      
      cv2.circle(plotScreen, (handPos[i][0], handPos[i][1]), 8, (0, 255, 0), 3)
      cv2.putText(plotScreen, handOpenText, (20, 200 + textHShift), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
      cv2.putText(plotScreen, fingerNumText, (20, 250 + textHShift), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
      textHShift += 100
    plotScreen = cv2.resize(plotScreen, RECORD_SIZE)
    cv2.imshow("screen", plotScreen)

    #record
    if Config.Record:
      frameRecoder.write(plotScreen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  if Config.Record:
    frameRecoder.release()
  capVideo.release()




  




  #





