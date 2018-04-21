from flask import Blueprint
from flask_restful import reqparse, abort, Api, Resource, request

import sys
sys.path.append('../')

from src.ai_models.rl_model import model

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

class test(Resource):
    def get(self):
        return 'hello, world'



class StateUpdater(Resource):

    def update_model(self, time_stamp, current_state, reward):
        model.add_memory(time_stamp, current_state)
        if time_stamp > 0:
            model.update_previous_memory(time_stamp - 1, reward=reward, next_state=current_state)

    def extractBodyFields(self):
        return int(request.form['reward']), int(request.form['time_stamp']), request.form['current_state']

    def post(self):
        reward, time_stamp, current_state = self.extractBodyFields()
        self.update_model(time_stamp, current_state, reward)
        return 201

class DirectionQuery(Resource):

    def extractArgs(self):
        args = request.args
        return int(args['time_stamp'])

    def get(self):
        time_stamp = self.extractArgs()
        action = model.get_action(time_stamp)
        return {'action': action}

api.add_resource(StateUpdater, '/update-state')
api.add_resource(DirectionQuery, '/query-direction')
