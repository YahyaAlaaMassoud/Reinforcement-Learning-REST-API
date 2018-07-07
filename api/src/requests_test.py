import requests
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

update_data = {'epsilon': json.dumps(0.93), 'epsilon_decay': json.dumps(0.9751), 'max_ep_len': json.dumps(50)}
r_update = requests.put('http://127.0.0.1:5000/api/init-params', data = update_data)
print(r_update.text)