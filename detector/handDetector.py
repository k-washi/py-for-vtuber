"""
Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces 
using Convolutional Neural Networks, https://github.com/victordibia/handtracking

"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#-----

import tensorflow as tf
import numpy as np
import math
import cv2

#-----
from detector.tracking import handTrackerCV

detection_graph = tf.Graph()
_score_thresh = 0.6

MODEL_ROOT = './model'
PATH_TO_CKPT = os.path.join(MODEL_ROOT,  'frozen_inference_graph.pb')

NUM_CLASS = 1

class handDetectorVtube():
  def __init__(self):
    self.detection_graph = None
    self.sess = None


  def load_inference_graph(self):
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    

  def detect_objects(self, image_np):
    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
    
    detection_boxes = self.detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    detection_scores = self.detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = self.detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = self.detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = self.sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

  def score_Classifier(self, boxes, scores, im_width, im_height, score_th = _score_thresh):
    bboxes = []
    slist = []
    for i, score in enumerate(scores):
      if score >= score_th:
        left = boxes[i][1] * im_width
        top =  boxes[i][0] * im_height

        bbox = (left, top, boxes[i][3] * im_width - left, boxes[i][2] * im_height - top) # (left, width, top, height)
        bboxes.append(bbox)
        slist.append(score)
    
    return bboxes, slist
  
  def handOpenDtector(self, mask):
    #開き具合(肌が見えている範囲で変わる)と開いた指の数を計算
    #指の数２以下をクローズ、3以上をオープンと認識したら安定する。

    #findcontours 輪郭抽出
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #max area
    arearatio = None
    l = 0
    if len(contours) > 0:
      cnt = max(contours, key = lambda x: cv2.contourArea(x))

      
      epsilon = 0.0005*cv2.arcLength(cnt,True) #arcLength 領域の周囲長さ
      

      #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
      hull = cv2.convexHull(cnt)
      areahull = cv2.contourArea(hull) #凸を考慮した面積
      areacnt = cv2.contourArea(cnt) #凸を考慮しない面積
      arearatio=((areahull-areacnt)/areacnt)*100

      
      approx= cv2.approxPolyDP(cnt,epsilon,True) #少ない点数に近似
      hull = cv2.convexHull(approx, returnPoints=False) #returnPoints 凸検出False
      defects = cv2.convexityDefects(approx, hull)

      #print("def")
      #print(defects)
      l = 0
      if defects is None:
        return arearatio, None

      #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html

      for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt= (100,180)
        
        fingerL = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) #指の間の距離
        hullL1 = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2) #指１の長さ
        hullL2 = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2) #指２の長さ
        s = (fingerL+hullL1+hullL2)/2
        ar = math.sqrt(s*(s-fingerL)*(s-hullL1)*(s-hullL2)) #ヘロンの公式
        
        d=(2*ar)/fingerL #凸の長さ
        
        #余弦定理
        angle = math.acos((hullL1**2 + hullL2**2 - fingerL**2)/(2*hullL1*hullL2)) * 180. /np.pi
  
        #無理な角度は無視
        if angle <= 90 and d>30:
            l += 1
        
      if l >= 1:
        l += 1
    
    return arearatio, l

   


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  from utils.captureVideo import cpatureVideo
  import cv2



  hd = handDetectorVtube()
  ht = handTrackerCV(kcf=True)

  fd = cpatureVideo(WIDTH=640, HEIGHT=480)
  WIDTH, HEIGHT = fd.get_size()

  fig, ax = plt.subplots()
  hd.load_inference_graph()
  

  while True:
  

    rgb_frame = fd.read()
    hsv_frame = fd.rgb2hsv(rgb_frame)

    boxes, scores = hd.detect_objects(rgb_frame)
    boxes, slist = hd.score_Classifier(boxes, scores, WIDTH, HEIGHT)

    mask = hsv_frame

    overlap = False
    counter = 0
    old_bbox = (0, 0, 0, 0)
    for box in boxes:

      counter += 1

      if counter == 1:
        overlap = False
        old_bbox = (box[0], box[1], box[2], box[3])

      elif counter >= 2:
        if old_bbox[0] < box[0] + box[2]/2 and box[0] + box[2]/2 < old_bbox[0] + old_bbox[2]:
          overlap = True
        else:
          overlap = False
        
      if not overlap:
        
        overlap = True
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False)
        ax.add_patch(rect)
        ht.setTrackingBox(hsv_frame, box[0], box[1], box[2], box[3])

      if counter >= 2:
        
        break
      
      

    tfs, boxes = ht.trackings(hsv_frame)
    for i, box in enumerate(boxes):
      if tfs[i]:

        handMask = fd.hsvSkinMasking(hsv_frame, box[0], box[1], box[2], box[3])
        handOpenRatio, finger_Num = hd.handOpenDtector(handMask)
        print(handOpenRatio)
        print(finger_Num)

        print(handOpenRatio)
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False)
        ax.add_patch(rect)

    plt.imshow(mask)
    plt.pause(0.01)
    plt.cla()

