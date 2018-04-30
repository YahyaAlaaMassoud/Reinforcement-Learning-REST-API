from flask import Flask

import sys
sys.path.append('./src/')
sys.path.append('./src/api/')
sys.path.append('./src/db/')
sys.path.append('./src/ai_models/')
sys.path.append('./src/api/resources/')

from api.api import api_bp

def create_app(config_filename):
    app = Flask(__name__)
    #app.config.from_object(config_filename)
    app.register_blueprint(api_bp, url_prefix = '/api')
    # from Model import db
    # db.init_app(app)
    return app
	
if __name__ == "__main__":
    app = create_app("config")
# host = '0.0.0.0' makes the server visible across the network (Extremly Visible Server)
    app.run(debug = True, host = '0.0.0.0')
