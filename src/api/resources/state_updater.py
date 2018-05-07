from flask_restful import Resource, request
from flask import jsonify
import numpy as np
import json

from ai_models.rl_model import Model
import db.rl_config as rl_config

class StateUpdater(Resource):
    
    model = Model()

    def update_model(self, time_stamp, current_state, reward, terminate):
        action = self.model.add_memory(time_stamp, current_state, terminate)
        if time_stamp > 0:
            self.model.update_previous_memory(time_stamp - 1, reward = reward, next_state = current_state)
        return action

    def extractBodyFields(self):
        return json.loads(request.form['reward']), json.loads(request.form['current_state']), json.loads(request.form['terminate'])

    def post(self):
        reward, current_state, terminate = self.extractBodyFields()
        time_stamp = rl_config.get_learning_param('time_stamp')
        action = self.update_model(time_stamp, current_state, reward, terminate)
        return jsonify({'time_stamp': time_stamp, 'action': action})