import tensorflow as tf
import numpy as np

import db.replay_memory as replay_memory

from pymongo.son_manipulator import SONManipulator

class DQN():
    def __init__(self, input_shape, output_units):
        self.input_shape = [-1]
        self.flattened_shape = 1
        for dim in input_shape:
            self.flattened_shape = self.flattened_shape * dim
            self.input_shape.append(dim)
#        print(self.flattened_shape)
        self.output_units = output_units
#        print(self.input_shape)
        self.create_model()
        self.init()
        self.save_model()
        
    def __hash__(self):
        return hash(self)
        
    def create_model(self):
        
        self.input_layer = tf.placeholder(shape = [None, self.flattened_shape], dtype = tf.float32)
        
        self.input = tf.reshape(self.input_layer, self.input_shape)

        self.dense1 = tf.layers.dense(inputs = self.input, units = 20, activation = tf.nn.relu)
        
        self.dense2 = tf.layers.dense(inputs = self.dense1, units = 10, activation = tf.nn.relu)

        self.Qout = tf.layers.dense(inputs = self.dense2, units = self.output_units, activation = None)
        
        self.prediction_value = tf.reduce_max(self.Qout, 1)
        
        self.prediction = tf.argmax(self.Qout, 1)
        
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        
        self.action = tf.placeholder(shape = [None], dtype = tf.int32)
        self.action_ont_hot = tf.one_hot(self.action, self.output_units, dtype = tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_ont_hot), axis = 1)
        
        self.temporal_difference = tf.square(self.targetQ - self.Q)
        
        self.loss = tf.reduce_mean(self.temporal_difference)
        
        self.optimizer = tf.train.AdamOptimizer(0.001)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        
        self.train_model = self.optimizer.minimize(self.loss)
        
    def init(self):
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        self.saver = tf.train.Saver()
    
    def get_prediction(self, state):
        self.load_model()
        return self.session.run(self.prediction, feed_dict = {self.input_layer : state}).tolist()
    
    def get_prediction_value(self, states):
        return self.session.run(self.prediction_value, feed_dict = {self.input_layer : states})
    
    def train_batch(self, states, targetQ, actions):
        return self.session.run(self.train_model, feed_dict = {
                                                                self.input_layer : states, \
                                                                self.targetQ : targetQ.tolist(), \
                                                                self.action : actions.tolist()
                                                              })
    
    def update_model(self, data, learning_rate):
        self.load_model()
        done = -(data['terminates'] - 1)
        target_rewards = data['rewards'] + learning_rate * done * self.get_prediction_value(data['next_states'])
        self.train_batch(data['current_states'], target_rewards, data['actions'])
        self.save_model()
        
    def save_model(self):
        self.save_path = self.saver.save(self.session, "/tmp/model.ckpt")
        
    def load_model(self):
        self.saver.restore(self.session, "/tmp/model.ckpt")
        
#dqn = DQN([4], 2)
#dqn.save_model()
#dqn.load_model()
#s = np.array([[1,0.5,0.3,0.4]])
#p = dqn.get_prediction(s)
#data = dqn.update_model(32)


