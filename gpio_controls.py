import RPi.GPIO as GPIO

from time import sleep

GPIO.setmode(GPIO.BOARD)

GPIO.setup(3, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)

GPIO.setup(37, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)
GPIO.setup(33, GPIO.OUT)

pwm=GPIO.PWM(7, 100)
pwm.start(0)

GPIO.output(3, True)
GPIO.output(5, False)

GPIO.output(35, True)
GPIO.output(33, False)

pwm.ChangeDutyCycle(90)

GPIO.output(7, True)

GPIO.output(37, True)

sleep(25)

GPIO.output(7, False)

GPIO.output(37, False)

GPIO.output(3, False)
GPIO.output(5, False)

GPIO.output(33, False)
GPIO.output(35, False)

pwm.stop()

GPIO.cleanup()
