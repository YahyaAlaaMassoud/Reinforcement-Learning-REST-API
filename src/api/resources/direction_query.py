from flask_restful import Resource, request

from ai_models.rl_model import Model

class DirectionQuery(Resource):
    
    model = Model()

    def extractArgs(self):
        args = request.args
        return int(args['time_stamp'])

    def get(self):
        time_stamp = self.extractArgs()
        action = self.model.get_action(time_stamp)
        return {'action': action}
