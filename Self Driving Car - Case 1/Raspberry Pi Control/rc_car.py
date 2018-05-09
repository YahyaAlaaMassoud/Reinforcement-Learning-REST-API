import RPi.GPIO as GPIO
import time

def init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    left = 2
    right = 3
    forward = 4

    GPIO.setup(left, GPIO.OUT, initial = False)
    GPIO.setup(right, GPIO.OUT, initial = False)
    GPIO.setup(forward, GPIO.OUT, initial = False)
    
    return left, right, forward

def clean():
    GPIO.cleanup()

def move_left(duration):
    left, right, forward = init()
    
    GPIO.output(left, 1)
    GPIO.output(forward, 1)
    
    time.sleep(duration)
    
    GPIO.output(left, 0)
    GPIO.output(forward, 0)
    
    clean()
    
def move_right(duration):
    left, right, forward = init()
    
    GPIO.output(right, 1)
    GPIO.output(forward, 1)
    
    time.sleep(duration)
    
    GPIO.output(right, 0)
    GPIO.output(forward, 0)
    
    print('move right')
    
    clean()
    
def move_forward(duration):
    _, _, forward = init()
    
    GPIO.output(forward, 1)
    
    time.sleep(duration)
    
    GPIO.output(forward, 0)
    
    clean()
    