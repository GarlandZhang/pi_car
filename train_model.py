from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--embeddings',  required=True, help='path to serialized db of facial embedding')
parser.add_argument('-r', '--recognizer', required=True, help='path to output model trained to recognize faces')
parser.add_argument('-l', '--le', required=True, help='path to output label encoder')
args = vars(parser.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings..")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels..")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training modedl...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

with open(args["recognizer"], "wb") as database:
  database.write(pickle.dumps(recognizer))

with open(args["le"], "wb") as label_file:
  label_file.write(pickle.dumps(le))


