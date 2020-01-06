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
from detector.trackering import trackingCV

detection_graph = tf.Graph()
_score_thresh = 0.3

MODEL_ROOT = './model'
PATH_TO_CKPT = os.path.join(MODEL_ROOT,  'frozen_inference_graph.pb')
PATH_TO_LABEL = os.path.join(MODEL_ROOT, 'hand_label_map.pbtxt')

NUM_CLASS = 1

class handDetector():
  def __init__(self):
    super().__init__()


  def load_inference_graph(self):
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess

  def detect_objects(self, image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
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
  fd = cpatureVideo(WIDTH=640, HEIGHT=480)
  WIDTH, HEIGHT = fd.get_size()

  fig, ax = plt.subplots()
  detection_graph, sess = hd.load_inference_graph()
  print("Capture")

  while True:
    rgb_frame = fd.read()
    print(np.shape(rgb_frame))
    boxes, scores = hd.detect_objects(rgb_frame, detection_graph, sess)
    boxes, slist = hd.score_Classifier(boxes, scores, WIDTH, HEIGHT)
    print(boxes)
    print(slist)
    for box in boxes:
      rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False)
      ax.add_patch(rect)

    plt.imshow(rgb_frame)
    plt.pause(0.01)
    plt.cla()

