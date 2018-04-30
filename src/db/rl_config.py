from pymongo import MongoClient
import numpy as np
import json
from flask import jsonify

client = MongoClient('mongodb://localhost:27017/')
db = client['rl-model']
rl_config = db['rl_config']

def init_params(input_shape, output_units, learning_rate = 0.99, epsilon_min = 0.01, epsilon = 1.0, epsilon_decay = 0.99991,\
                rand_steps = 10000 , update_freq = 4, max_ep_len = 200, batch_size = 32):
    if check_data():
        return False
    rl_config.insert({
                      'input_shape': json.dumps(input_shape),
                      'output_units': json.dumps(output_units),
                      'learning_rate': json.dumps(learning_rate), 
                      'epsilon_min': json.dumps(epsilon_min), 
                      'epsilon': json.dumps(epsilon), 
                      'epsilon_decay': json.dumps(epsilon_decay), 
                      'rand_steps': json.dumps(rand_steps),
                      'update_freq': json.dumps(update_freq), 
                      'max_ep_len': json.dumps(max_ep_len),
                      'episode_number': json.dumps(0),
                      'time_stamp': json.dumps(0),
                      'batch_size': json.dumps(batch_size)
                     })
    return True

def get_learning_param(param):
    try:
        param_value = list(rl_config.find({}, {'_id': 0}))[0][param]
        return json.loads(param_value)
    except Exception as ex:
        print('No param found with this name ' + str(ex))
        return False

def update_learning_param(param, new_value):
    param_value = get_learning_param(param)
    if param_value is False:
        return False
    rl_config.find_one_and_update({param : json.dumps(param_value)},
                                  {"$set": {param : json.dumps(new_value)}})
    return True
    
def check_data():
    data = list(rl_config.find({}, {'_id': 0}))
    if len(data):
        return True
    return False

#init_params()
#print(get_learning_param('learning_rate'))
#update_learning_param('learning_rate', 0.99)
#print(get_learning_param('learning_rate'))
