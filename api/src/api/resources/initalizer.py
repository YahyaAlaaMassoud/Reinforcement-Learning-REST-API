from flask_restful import Resource, request
from flask import jsonify
import json

import db.rl_config as rl_config

class Initalizer(Resource):
    
    def post(self, learning_rate = 0.99, epsilon_min = 0.01, epsilon = 1.0, epsilon_decay = 0.99861,\
                   rand_steps = 70 , update_freq = 4, max_ep_len = 200, batch_size = 64):
        input_shape   = json.loads(request.form['input_shape'])
        output_units  = json.loads(request.form['output_units'])
        learning_rate = json.loads(request.form.get('learning_rate') or str(learning_rate))
        epsilon_min   = json.loads(request.form.get('epsilon_min') or str(epsilon_min))
        epsilon       = json.loads(request.form.get('epsilon') or str(epsilon))
        epsilon_decay = json.loads(request.form.get('epsilon_decay') or str(epsilon_decay))
        rand_steps    = json.loads(request.form.get('rand_steps') or str(rand_steps))
        update_freq   = json.loads(request.form.get('update_freq') or str(update_freq))
        max_ep_len    = json.loads(request.form.get('max_ep_len') or str(max_ep_len))
        batch_size    = json.loads(request.form.get('batch_size') or str(batch_size))
        res = rl_config.init_params(
                                      input_shape = input_shape,
                                      output_units = output_units,
                                      learning_rate = learning_rate, 
                                      epsilon_min = epsilon_min,
                                      epsilon = epsilon,
                                      epsilon_decay = epsilon_decay,
                                      rand_steps = rand_steps,
                                      update_freq = update_freq,
                                      max_ep_len = max_ep_len,
                                      batch_size = batch_size
                                   )
        return jsonify(success = res)

    def put(self):
        param_names = list(request.form)
        res = []
        for param_name in param_names:
            res.append(rl_config.update_learning_param(param_name, json.loads(request.form[param_name])))
        return jsonify(success = res)
    
