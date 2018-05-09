import requests
import json

def init_params(input_shape, output_units):
    init_data = {'input_shape': json.dumps(input_shape), 'output_units': json.dumps(output_units)}
    r_init = requests.post('http://172.20.10.4:5000/api/init-params', data = init_data)
    return json.loads(r_init.text)['success']

def get_action(state, reward, terminate):
    data = {'current_state': json.dumps(state), 'reward': json.dumps(reward), 'terminate': json.dumps(terminate)}
    r_post = requests.post('http://172.20.10.4:5000/api/update-state', data = data)
    return json.loads(r_post.text)['action'][0]