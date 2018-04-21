# import sys
# sys.path.append('../')

import src.db.rl_model as replay_memory

class Model:

    def get_action(self, time_stamp):
        #print 'R'
        return 'R'
        #return replay_memory.query_action(time_stamp)

    def add_memory(self, time_stamp, current_state):
        action = self.create_action(time_stamp, current_state)
        print (action)
        replay_memory.insert_transition(time_stamp, current_state, action)

    def create_action (self, time_stamp, current_state):
        return 'R'

    def update_previous_memory(self, time_stamp, reward=None, next_state=None):
      replay_memory.update_transition(time_stamp, reward, next_state)

model = Model()
