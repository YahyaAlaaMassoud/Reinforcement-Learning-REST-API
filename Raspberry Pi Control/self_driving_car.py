import requests
import json
from ultrasonic import get_distance
from rc_car import move_left, move_right, move_forward
from camera import capture
from reward import get_reward
import time

#r1 = requests.get('http://192.168.137.1:5000/api/')
#print(r1.text)

for i in range(20):
    d = get_distance()
    print(d)
    print(get_reward(d, 60))
    #move_forward(0.2)

    im = capture()
    print(im.shape)
    print()
    print()
    