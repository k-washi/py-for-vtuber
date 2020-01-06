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

#-----
from detector.tracking import handTracker

detection_graph = tf.Graph()
_score_thresh = 0.3

MODEL_ROOT = './model'
PATH_TO_CKPT = os.path.join(MODEL_ROOT,  'frozen_inference_graph.pb')

NUM_CLASS = 1

class handDetector():
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


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  from utils.captureVideo import cpatureVideo
  import cv2



  hd = handDetector()
  ht = handTracker()

  fd = cpatureVideo(WIDTH=640, HEIGHT=480)
  WIDTH, HEIGHT = fd.get_size()

  fig, ax = plt.subplots()
  hd.load_inference_graph()
  

  while True:
  

    rgb_frame = fd.read()
    hsv_frame = fd.rgb2hsv(rgb_frame)
    boxes, scores = hd.detect_objects(rgb_frame)
    boxes, slist = hd.score_Classifier(boxes, scores, WIDTH, HEIGHT)
    #print(boxes, scores)
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
        #rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False)
        #ax.add_patch(rect)
        ht.setTrackingBox(rgb_frame, box[0], box[1], box[2], box[3])

        

      if counter >= 2:
        
        break
      
      

    tfs, bboxes = ht.trackings(rgb_frame)
    #print(tfs)
    for tf, box in zip(tfs, boxes):
      if tf:
        #h, s, v = fd.hsvExtraction(hsv_frame, box[0], box[1], box[2], box[3])
        print(box)
        mask = fd.hsvMasking(hsv_frame, box[0], box[1], box[2], box[3])

        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False)
        ax.add_patch(rect)

    plt.imshow(rgb_frame)
    plt.pause(0.01)
    plt.cla()

