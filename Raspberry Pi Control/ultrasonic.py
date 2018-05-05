import RPi.GPIO as GPIO
import time

def init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    ECHO = 14
    TRIG = 15

    GPIO.setup(TRIG, GPIO.OUT, initial = False)
    GPIO.setup(ECHO, GPIO.IN)
    return ECHO, TRIG

def clean():
    GPIO.cleanup()

def get_distance():
    ECHO, TRIG = init()
    
    GPIO.output(TRIG, True)
    time.sleep(0.00000001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
            
    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration * 17150

    distance = round(distance, 2)

    clean()
    
    return distance
