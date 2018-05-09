from flask_restful import Resource

class test(Resource):
    def put(self):
        return 'put, yahya'
    
    def delete(self):
        return 'delete, yahya'
    
    def post(self):
        return 'post, yahya' 
    
    def get(self):
        return 'get, yahya'
