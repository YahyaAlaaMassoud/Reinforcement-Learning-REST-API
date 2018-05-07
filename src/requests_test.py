import requests
import numpy as np
import json

init_data = {'input_shape': json.dumps([4]), 'output_units': json.dumps(2), 'rand_steps': json.dumps(0), 'epsilon': json.dumps(1.0), 'epsilon_decay': json.dumps(0.9)}
r_init = requests.post('http://127.0.0.1:5000/api/init-params', data = init_data)
print(r_init.text)

data = {'current_state':json.dumps([[1,1,1,1]]), 'reward':json.dumps(2), 'terminate':json.dumps(0)}
r1 = requests.post('http://127.0.0.1:5000/api/update-state', data = data)
print(r1.text)

data = {'current_state':json.dumps([3,1,3,1]), 'reward':json.dumps(2), 'terminate':json.dumps(0)}
r1 = requests.post('http://127.0.0.1:5000/api/update-state', data = data)
print(r1.text)

update_data = {'time_stamp': json.dumps(0), 'update_freq': json.dumps(4), 'epsilon': json.dumps(1.0), 'epsilon_decay': json.dumps(0.9981), 'rand_steps': json.dumps(2000)}
r_update = requests.put('http://127.0.0.1:5000/api/init-params', data = update_data)
print(r_update.text)