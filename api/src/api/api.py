from flask import Blueprint
from flask_restful import Api

from resources.state_updater import StateUpdater
from resources.test_resource import test
from resources.direction_query import DirectionQuery
from resources.initalizer import Initalizer

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(test, '/')
api.add_resource(Initalizer, '/init-params')
api.add_resource(StateUpdater, '/update-state')
api.add_resource(DirectionQuery, '/query-direction')
