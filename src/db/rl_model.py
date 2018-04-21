import sys
sys.path.append('../')

from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['rl-model']
replay_memory = db['replay_memory']


def insert_transition(time_stamp, current_state, action):
    replay_memory.insert({'time_stamp': time_stamp,
            'current_state': current_state,
            'action': action}
    )

def update_transition(time_stamp, reward, next_state):
    replay_memory.find_one_and_update({'time_stamp': time_stamp},
                                    {"$set": {'reward': reward, 'next_state': next_state}})
def query_action(time_stamp):
    transition = replay_memory.find_one({'time_stamp': time_stamp})
    return transition['action']

def clear_replay_memory():
    replay_memory.remove_many({})

def print_replay_memory():
    for mem in replay_memory.find():
           print (mem)


#if __name__ == '__main__':
    #insert_transition(134, 10101, 'R')
    #update_transition(134, 13445, 134)
    #print query_action(134)
    #replay_memory.delete_many({})
    #print_replay_memory():
