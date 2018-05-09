from pymongo import MongoClient
import numpy as np
import random
import json

client = MongoClient('mongodb://localhost:27017/')
db = client['rl-model']
replay_memory = db['replay_memory']

def insert_transition(time_stamp, current_state, action, terminate):
    replay_memory.insert({
                            'time_stamp': json.dumps(time_stamp),
                            'current_state': json.dumps(current_state),
                            'action': json.dumps(action),
                            'terminate': json.dumps(terminate)
                         })

def update_transition(time_stamp, reward, next_state):
    replay_memory.find_one_and_update({'time_stamp': json.dumps(time_stamp)},
                                      {"$set": {'reward': json.dumps(reward), 'next_state': json.dumps(next_state)}})
    
def query_action(time_stamp):
#    print(time_stamp)
    transition = replay_memory.find_one({'time_stamp': json.dumps(time_stamp)})
#    print(transition)
    return json.loads(transition['action'])

def clear_replay_memory():
    replay_memory.delete_many({})
    
def sample_experiences(sample_size):
    all_experiences = list(replay_memory.find({}, {'_id': 0}))
    sample = random.sample(all_experiences, min(len(all_experiences), sample_size))
    current_states = np.vstack(np.array([json.loads(mem['current_state']) for mem in sample if 'next_state' and 'reward' in mem]))
    next_states = np.vstack(np.array([json.loads(mem['next_state']) for mem in sample if 'next_state' and 'reward' in mem]))
    actions = np.vstack(np.array([json.loads(mem['action']) for mem in sample if 'next_state' and 'reward' in mem]))
    rewards = np.vstack(np.array([json.loads(mem['reward']) for mem in sample if 'next_state' and 'reward' in mem]))
    terminates = np.vstack(np.array([json.loads(mem['terminate']) for mem in sample if 'next_state' and 'reward' in mem]))
    return { 
             'current_states' : current_states, 
             'next_states' : next_states,
             'actions' : actions.reshape(actions.shape[0],),
             'rewards' : rewards.reshape(rewards.shape[0],),
             'terminates' : terminates.reshape(terminates.shape[0],)
           }
    
def print_replay_memory():
    for mem in replay_memory.find():
        print(mem['next_state'])
        
#di = sample_experiences(10)

#clear_replay_memory()
#
#done = -(di['terminates'] - 1)
#target_rewards = di['rewards'] + 0.99 * done * np.random.randn(15, 1)
#print(target_rewards.tolist())
#print(di['actions'].reshape(di['actions'].shape[0],).tolist())
#print(di['rewards'].reshape(di['rewards'].shape[0],).tolist())
#print(di['next_states'].shape)
#print(di['current_states'].shape)

#insert transitions manually
#for i in range(64):
#    insert_transition(i, np.random.randn(1, 4).tolist(), np.random.randint(0, 3), False)
#    if i > 0:
#        update_transition(i - 1, i / 64, np.random.randn(1, 4).tolist())

#if __name__ == '__main__':
    #insert_transition(134, 10101, 'R')
    #update_transition(134, 13445, 134)
    #print query_action(134)
    #replay_memory.delete_many({})
    #print_replay_memory():
