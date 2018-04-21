from flask import Flask

import sys
sys.path.append('../')

from src.api.api import api_bp


def create_app(config_filename):
    app = Flask(__name__)
    #app.config.from_object(config_filename)
    app.register_blueprint(api_bp, url_prefix='/api')
    # from Model import db
    # db.init_app(app)
    return app
if __name__ == "__main__":
    app = create_app("config")
    app.run(debug=True)