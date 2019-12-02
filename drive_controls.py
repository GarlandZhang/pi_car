from recognize import Recognizer

import RPi.GPIO as GPIO
import picamera
from time import sleep
import cv2
import numpy as np
import io

args = {
  'embedding_model': 'openface_nn4.small2.v1.t7',
  'detector': 'face_detection_model',
  'le': 'output/le.pickle',
  'recognizer': 'output/recognizer.pickle',
  'confidence': 0.3,
}

camera = picamera.PiCamera()

rec = Recognizer(args)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)

GPIO.setup(37, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)
GPIO.setup(33, GPIO.OUT)

pwm = GPIO.PWM(7, 100)
pwm.start(0)

while True:
  stream = io.BytesIO()
  camera.capture(stream, format='jpeg')
  image_data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
  image = cv2.imdecode(image_data, 1)
  args['image'] = image

  image, detection, class_confidence, class_name = rec.recognize_face(args)
  if detection is not None:

    # go one direction

    GPIO.output(3, True)
    GPIO.output(5, False)

    GPIO.output(35, True)
    GPIO.output(33, False)

    pwm.ChangeDutyCycle(100)

    GPIO.output(7, True)
    GPIO.output(37, True)

    sleep(5)

    # go other direction
    GPIO.output(7, False)
    GPIO.output(37, False)

    GPIO.output(3, False)
    GPIO.output(5, True)

    GPIO.output(35, False)
    GPIO.output(33, True)

    pwm.ChangeDutyCycle(100)

    GPIO.output(7, True)
    GPIO.output(37, True)

    sleep(5)   


    # zero out everything

    GPIO.output(7, False)
    GPIO.output(37, False)

    GPIO.output(3, False)
    GPIO.output(5, False)

    GPIO.output(35, False)
    GPIO.output(33, False)

  cv2.imshow('image', image)
  if cv2.waitKey(30) & 0xFF == ord('q'):
    break

pwm.stop() 
GPIO.cleanup()
cv2.destroyAllWindows()
