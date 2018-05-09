import db.replay_memory as replay_memory
import db.rl_config as rl_config
import numpy as np

from ai_models.q_network import DQN

class Model:
    
    if rl_config.check_data() is True:
        model = DQN(rl_config.get_learning_param('input_shape'), rl_config.get_learning_param('output_units'))
        print('model init')
    
    def get_action(self, time_stamp):
        return replay_memory.query_action(time_stamp)

    def add_memory(self, time_stamp, current_state, terminate):
        action = self.create_action(time_stamp, current_state)
        self.check_model_update()
        self.increment_timestamp(time_stamp)
        replay_memory.insert_transition(time_stamp, current_state, action, terminate)
        return action
    
    def increment_timestamp(self, time_stamp):
        rl_config.update_learning_param('time_stamp', time_stamp + 1)

    def create_action (self, time_stamp, current_state):
        epsilon = rl_config.get_learning_param('epsilon')
        action_space = rl_config.get_learning_param('output_units')
        time_stamp = rl_config.get_learning_param('time_stamp')
        rand_steps = rl_config.get_learning_param('rand_steps')
        
        if (np.random.rand() < epsilon) or (time_stamp < rand_steps):
            action = [np.random.randint(0, action_space)]
            print('took random action')
        else:
            action = self.model.get_prediction(np.array(current_state))
            print('took a wise action')
        print(epsilon)
        return action
    
    def check_model_update(self):
        time_stamp = rl_config.get_learning_param('time_stamp')
        update_freq = rl_config.get_learning_param('update_freq')
        rand_steps = rl_config.get_learning_param('rand_steps')
        
        if time_stamp % update_freq == 0 and time_stamp > rand_steps:
            epsilon = rl_config.get_learning_param('epsilon')
            epsilon_decay = rl_config.get_learning_param('epsilon_decay')
            epsilon_min = rl_config.get_learning_param('epsilon_min')
            batch_size = rl_config.get_learning_param('batch_size')
            learning_rate = rl_config.get_learning_param('learning_rate')

            print('update model')            
            data = replay_memory.sample_experiences(batch_size)
            self.model.update_model(data, learning_rate)
            
            if epsilon >= epsilon_min:
                rl_config.update_learning_param('epsilon', epsilon * epsilon_decay)

    def update_previous_memory(self, time_stamp, reward = None, next_state = None):
        replay_memory.update_transition(time_stamp, reward, next_state)
