import requests
import json
import gym
import numpy as np

# OpenAI CartPole game enviroment
env = gym.make('CartPole-v1')

state = env.reset()
reward = 0
terminate = 0

# Init initial parameters for Q-learning algorithm
init_data = {'input_shape': json.dumps([4]), 'output_units': json.dumps(2), 'rand_steps': json.dumps(2000), 'epsilon': json.dumps(1.0), 'epsilon_decay': json.dumps(0.9981)}
r_init = requests.post('http://127.0.0.1:5000/api/init-params', data = init_data)
print(r_init.text)

# Get action from neural network to the current state
def get_action(state, reward, terminate):
    data = {'current_state': json.dumps(state), 'reward': json.dumps(reward), 'terminate': json.dumps(terminate)}
    r_post = requests.post('http://127.0.0.1:5000/api/update-state', data = data)
    return json.loads(r_post.text)['action'][0]

acum_reward = 0
# Iterate to make the agent a better learner
for i in range(500):
    # get action by doing and API call
    a = get_action([state.tolist()], reward, terminate)
    # perform the step got from API into the enviroment
    state, reward, terminate, _ = env.step(a)
    
    if terminate == False:
        acum_reward += reward
    else:
        print(acum_reward)
        acum_reward = 0
        state = env.reset()
        reward = 0
        terminate = 0
        continue
