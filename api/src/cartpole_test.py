import gym
import json
import requests

env = gym.make('CartPole-v1')

s = env.reset()
r = 0
t = 0

cum_r = 0
all_r = []
max_ep = 50
cnt = 0
eps = 700
ep = 0
while ep < eps:
    try:
        data = {'current_state':json.dumps(s.reshape(1, 4).tolist()), 'reward':json.dumps(r), 'terminate':json.dumps(t)}
        request = requests.post('http://127.0.0.1:5000/api/update-state', data = data)
        action = json.loads(request.text)['action'][0]
        s, r, t, _ = env.step(action)
        if t is True:
            r = -1.0
        else:
            r = 1.0
        cum_r += r
        cnt += 1
        if t is True or cnt == max_ep:
            print(str(ep) + ' : ' + str(cum_r))
            cum_r = 0
            cnt = 0
            ep += 1
            s = env.reset()
    except Exception as ex:
        print(ex)