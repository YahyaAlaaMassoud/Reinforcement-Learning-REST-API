
class Model:

    def get_action(self, time_stamp):
        return 'R'

    def create_action (self, time_stamp, current_state):
        return 'R'

    def add_memory(self, time_stamp, current_state):
        action = self.create_action(time_stamp, current_state)
        #database insert

    def update_memory(self, time_stamp, reward=None, next_state=None):
        if reward:
            print 'update reward in db'
            #database update
        if next_state:
            print 'update reward in db'
            #database update

model = Model()
