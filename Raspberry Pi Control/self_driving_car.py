import requests
import json
from ultrasonic import get_distance
from rc_car import move_left, move_right, move_forward
from camera import capture
from reward import get_reward
from server import init_params, get_action
import time

r_init = init_params([30 * 30], 3)

for i in range(0):
    st = time.time()
    
    distance = 401
    while distance >= 400 and distance <= 3000:
        distance = get_distance()
    reward = get_reward(distance, 60)
    print(distance, reward)
    terminate = False
    if distance < 60:
        terminate = True

    state = capture()
    print(state.shape)
    action = get_action(state.tolist(), reward, terminate)
    print(action)
    
    duration = 0.5
    
    if distance > 60:
        if action == 0:
            move_left(duration)
        elif action == 1:
            move_right(duration)
    else:
        input('do you want me to continue?')
        
        
    print('took: ' + str(time.time() - st))
    