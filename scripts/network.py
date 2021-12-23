import os
import re
import numpy as np
import tensorflow as tf
import time

class Actions():
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class NetworkVPCore(object):
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                vars = tf.global_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    
    def _create_graph_inputs(self):
        self.x = tf.placeholder(tf.float32, [None, Config.NN_INPUT_SIZE], name='X')
 
    def _create_graph_outputs(self):
        # FCN
        self.fc1 = tf.layers.dense(inputs=self.final_flat, units = 256, use_bias = True, activation=tf.nn.relu, name = 'fullyconnected1')

        # Cost: p
        self.logits_p = tf.layers.dense(inputs = self.fc1, units = self.num_actions, name = 'logits_p', activation = None)
        self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)

        # Cost: v 
        self.logits_v = tf.squeeze(tf.layers.dense(inputs=self.fc1, units = 1, use_bias = True, activation=None, name = 'logits_v'), axis=[1])

    def predict_p(self, x):
        return self.sess.run(self.softmax_p, feed_dict={self.x: x})

    def predict_v(self, x):
        return self.sess.run(self.logits_v, feed_dict={self.x: x})

    def get_lstm_output(self, x):
        return self.sess.run(self.rnn_output, feed_dict={self.x: x})

    def predict_p_from_lstm_output(self, host_agent_vec, lstm_output):
        return self.sess.run(nn.softmax_p, feed_dict={self.host_agent_vec: host_agent_vec, self.layer1_input: layer1_input})

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
        self.saver.restore(self.sess, filename)

class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super(self.__class__, self).__init__(device, model_name, num_actions)

    def _create_graph(self):
        # Use shared parent class to construct graph inputs
        self._create_graph_inputs()

        if Config.USE_REGULARIZATION:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
        else:
            regularizer = None

        if Config.NORMALIZE_INPUT:
            self.avg_vec = tf.constant(Config.NN_INPUT_AVG_VECTOR, dtype = tf.float32)
            self.std_vec = tf.constant(Config.NN_INPUT_STD_VECTOR, dtype = tf.float32)
            self.x_normalized = (self.x - self.avg_vec) / self.std_vec
        else:
            self.x_normalized = self.x


        if Config.MULTI_AGENT_ARCH == 'RNN':
            num_hidden = 64
            max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED
            self.num_other_agents = self.x[:,0]
            self.host_agent_vec = self.x_normalized[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
            self.other_agent_vec = self.x_normalized[:,Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
            self.other_agent_seq = tf.reshape(self.other_agent_vec, [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(num_hidden), self.other_agent_seq, dtype=tf.float32, sequence_length=self.num_other_agents)
            self.rnn_output = self.rnn_state.h
            self.layer1_input = tf.concat([self.host_agent_vec, self.rnn_output],1, name='layer1_input')
            self.layer1 = tf.layers.dense(inputs=self.layer1_input, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer1')

        self.layer2 = tf.layers.dense(inputs=self.layer1, units=256, activation=tf.nn.relu, name = 'layer2')
        self.final_flat = tf.contrib.layers.flatten(self.layer2)
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    USE_REGULARIZATION  = True
    ROBOT_MODE          = True
    EVALUATE_MODE       = True

    SENSING_HORIZON     = 8.0

    MIN_POLICY = 1e-4

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 20
    MULTI_AGENT_ARCH = 'RNN'

    DEVICE                        = '/cpu:0' # Device

    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius, 
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.tile(OTHER_AGENT_AVG_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.tile(OTHER_AGENT_STD_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])
            
    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH



if __name__ == '__main__':
    actions = Actions().actions
    num_actions = Actions().num_actions
    nn = NetworkVP_rnn(Config.DEVICE, 'network', num_actions)
    nn.simple_load()

    obs = np.zeros((Config.FULL_STATE_LENGTH))
    obs = np.expand_dims(obs, axis=0)

    num_queries = 10000
    t_start = time.time()
    for i in range(num_queries):
        obs[0,0] = 10 # num other agents
        obs[0,1] = np.random.uniform(0.5, 10.0) # dist to goal
        obs[0,2] = np.random.uniform(-np.pi, np.pi) # heading to goal
        obs[0,3] = np.random.uniform(0.2, 2.0) # pref speed
        obs[0,4] = np.random.uniform(0.2, 1.5) # radius
        predictions = nn.predict_p(obs, None)[0]
    t_end = time.time()
    print("avg query time:", (t_end - t_start)/num_queries)
    print("total time:", t_end - t_start)
    # action = actions[np.argmax(predictions)]
    # print "action:", action
