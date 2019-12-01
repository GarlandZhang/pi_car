import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

class Recognizer:
  def __init__(self, args):
      # load face detector
      print("[INFO] loading face detector...")
      proto_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
      model_path = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
      self.detector = cv2.cv2.dnn.readNetFromCaffe(proto_path, model_path)

      # load face embedding model
      print("[INFO] loading face recognizer...")
      self.embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

      # load actual face recognition model along with label encoder
      self.recognizer = pickle.loads(open(args["recognizer"], "rb").read())
      self.le = pickle.loads(open(args["le"], "rb").read())


  def recognize_face(self, args):
    detector = self.detector
    embedder = self.embedder
    recognizer = self.recognizer
    le = self.le    

    # load image
    image = args['image']
    if isinstance(image, str):
      image = cv2.imread(image)
    image = imutils.resize(image, width=600)
    height, width, _ = image.shape

    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # detect
    detector.setInput(image_blob)
    detections = detector.forward()

    size_min = 20

    largest_detection = None

    num_detections = detections.shape[2]
    for i in range(0, num_detections):
      detection = detections[0, 0, i]
      confidence = detection[2]
      if confidence > args['confidence']:
        print('detection: {}'.format(detection))
        if largest_detection is None or largest_detection[2] < confidence:
          largest_detection = detection
        
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x0, y0, x1, y1 = box.astype("int")
        
        # extract face ROI
        face = image[y0:y1, x0:x1]
        face_height, face_width, _ = face.shape

        # ignore any face too small
        if face_width < size_min or face_height < size_min:
          continue

        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        embedding = embedder.forward()

        # perform classif. to recognize face
        preds = recognizer.predict_proba(embedding)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # draw bounding box of face along with probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = y0 - 10 if y0 - 10 > 10 else y0 + 10
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return image, largest_detection, proba, name

    # show output image
    # cv2.imwrite("output.jpg", image)

    return image, largest_detection, None, None 

def recognize_face(args):
  # load face detector
  print("[INFO] loading face detector...")
  proto_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
  model_path = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
  detector = cv2.cv2.dnn.readNetFromCaffe(proto_path, model_path)

  # load face embedding model
  print("[INFO] loading face recognizer...")
  embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

  # load actual face recognition model along with label encoder
  recognizer = pickle.loads(open(args["recognizer"], "rb").read())
  le = pickle.loads(open(args["le"], "rb").read())

  # load image
  image = cv2.imread(args["image"])
  image = imutils.resize(image, width=600)
  height, width, _ = image.shape

  image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

  # detect
  detector.setInput(image_blob)
  detections = detector.forward()

  size_min = 20

  largest_detection = None

  num_detections = detections.shape[2]
  for i in range(0, num_detections):
    detection = detections[0, 0, i]
    confidence = detection[2]
    if confidence > args['confidence']:
      if largest_detection is None or largest_detection[2] < confidence:
        largest_detection = detection
      
      box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
      x0, y0, x1, y1 = box.astype("int")
      
      # extract face ROI
      face = image[y0:y1, x0:x1]
      face_height, face_width, _ = face.shape

      # ignore any face too small
      if face_width < size_min or face_height < size_min:
        continue

      face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
      embedder.setInput(face_blob)
      embedding = embedder.forward()

      # perform classif. to recognize face
      preds = recognizer.predict_proba(embedding)[0]
      j = np.argmax(preds)
      proba = preds[j]
      name = le.classes_[j]

      # draw bounding box of face along with probability
      text = "{}: {:.2f}%".format(name, proba * 100)
      y = y0 - 10 if y0 - 10 > 10 else y0 + 10
      cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
      cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

      return image, largest_detection, proba, name

  # show output image
  cv2.imwrite("output.jpg", image)

  return image, largest_detection, None, None 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", required=True, help="path to input image")
  parser.add_argument("-d", "--detector", required=True, help="path to OpenCVs deep learning face detector")
  parser.add_argument("-m", "--embedding-model", required=True, help="path to OPenCVs deep learning face embedding model")
  parser.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
  parser.add_argument("-l", "--le", required=True, help="path to label encoder")
  parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
  args = vars(parser.parse_args())

  recognize_face(args)
