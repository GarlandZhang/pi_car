from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataset', required=True, help='path to input directory of faces + images')
parser.add_argument('-e', '--embeddings', required=True, help='path to output serialized db of facial embeddings')
parser.add_argument('-d', '--detector', required=True, help='path to OpenCVs deep learning face detector')
parser.add_argument('-m', '--embedding-model', required=True, help='path to OpenCVs deep learning face embedding model')
parser.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
args = vars(parser.parse_args())

# load serialized face detector from disk
print("[INFO] loading face detector...")
proto_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
model_path = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

print("[INFO] loading face detector...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab paths to input images in our dataset
print("[INFO] quantifying faces...")
image_paths = list(paths.list_images(args["dataset"]))

# initialize our lists of extracted facial embeddings and corresponding people names
known_embeddings = []
known_names = []
total_faces = 0

size_min = 20

for i, image_path in enumerate(image_paths):
  # extract the person name from image path
  print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
  name = image_path.split(os.path.sep)[-2]

  # load image,resize to have width of 600 px, and grab image dims
  image = cv2.imread(image_path)
  image = imutils.resize(image, width=600)
  height, width, _ = image.shape

  # construct blob from image
  image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
  
  # apply opencvs deep learning-based face detector to localize faces in input image
  detector.setInput(image_blob)
  detections = detector.forward()

  if len(detections) > 0:
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
      box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
      x0, y0, x1, y1 = box.astype('int')

      # extract face ROI
      face = image[y0:y1, x0:x1]
      face_height, face_width, _ = face.shape

      if face_width < size_min or face_height < size_min:
        continue

      # construct blob for face ROI to pass to embedding model
      face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
      embedder.setInput(face_blob)
      embedding = embedder.forward()

      known_names.append(name)
      known_embeddings.append(embedding.flatten())
      total_faces += 1

# dump facial embeddings and names
print("[INFO] serializing {} encoddings...".format(total_faces))
data = {"embeddings": known_embeddings, "names": known_names}

print(data)

with open(args["embeddings"], "wb") as embedding_file:
  embedding_file.write(pickle.dumps(data))


