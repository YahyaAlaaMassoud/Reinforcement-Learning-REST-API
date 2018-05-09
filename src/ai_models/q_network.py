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
        self.create_model_ANN()
        self.init()
        
        try:
            self.load_model()
        except Exception as ex:
            print(ex)
            self.save_model()
        
    def __hash__(self):
        return hash(self)
        
    def create_model_ANN(self):
        
        self.input_layer = tf.placeholder(shape = [None, self.flattened_shape], dtype = tf.float32)
        
        self.input = tf.reshape(self.input_layer, self.input_shape)

        self.dense1 = tf.layers.dense(inputs = self.input, units = 200, activation = tf.nn.relu)
        
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
        
    def create_model_CNN(self):
        #INPUT [20,20,1] OUTPUT 4
        #RESHAPE input as (1, 20*20*1)
        self.input_layer = tf.placeholder(shape = [None, self.flattened_shape], dtype = tf.float32)
        
        self.input_image = tf.reshape(self.input_layer, self.input_shape)
        
        #self.conv1 = tf.layers.conv2d(inputs = self.input_image, filters = 32, kernel_size = 8, strides = 4, padding = "valid", activation = tf.nn.relu)

        #self.conv2 = tf.layers.conv2d(inputs = self.conv1, filters = 64, kernel_size = 4, strides = 2, padding = "valid", activation = tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs = self.input_image, filters = 64, kernel_size = 4, strides = 2, padding = "valid", activation = tf.nn.relu)

        self.conv3 = tf.layers.conv2d(inputs = self.conv2, filters = 64, kernel_size = 3, strides = 1, padding = "valid", activation = tf.nn.relu)

        self.conv4 = tf.layers.conv2d(inputs = self.conv3, filters = 512, kernel_size = 7, strides = 1, padding = "valid", activation = tf.nn.relu)

        self.new_shape = self.conv4.shape[1] * self.conv4.shape[2] * self.conv4.shape[3]

        self.conv4_flat = tf.reshape(self.conv4, [-1, self.new_shape])

        self.dense = tf.layers.dense(inputs = self.conv4_flat, units = 256, activation = tf.nn.relu)

        self.Qout = tf.layers.dense(inputs = self.dense, units = self.output_units, activation = None)
        
        self.prediction_value = tf.reduce_max(self.Qout, 1)
        
        self.prediction = tf.argmax(self.Qout, 1)
        
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        
        self.action = tf.placeholder(shape = [None], dtype = tf.int32)
        self.action_ont_hot = tf.one_hot(self.action, self.output_units, dtype = tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_ont_hot), axis = 1)
        
        self.temporal_difference = tf.square(self.targetQ - self.Q)
        
        self.loss = tf.reduce_mean(self.temporal_difference)
        
        self.optimizer = tf.train.AdamOptimizer(0.001)
        
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
        self.save_path = self.saver.save(self.session, "src/ai_models/tmp/model")
        
    def load_model(self):
#        self.saver.restore(self.session, "src/ai_models/tmp/model.ckpt")
        saver = tf.train.import_meta_graph('src/ai_models/tmp/model.meta')
        saver.restore(self.session, tf.train.latest_checkpoint('src/ai_models/tmp/'))
        
#dqn = DQN([4], 2)
#dqn.save_model()
#dqn.load_model()
#s = np.array([[1,0.5,0.3,0.4]])
#p = dqn.get_prediction(s)
#data = dqn.update_model(32)


