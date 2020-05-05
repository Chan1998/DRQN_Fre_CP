import tensorflow as tf
import numpy as np
from collections import deque

#build DQN_agent
class DeepQNetwork():
    def __init__(self, learning_rate, state_size, action_size, step_size,
                 hidden_size, sess=None, gamma=0.9, epsilon=0.8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.state_size =  state_size
        self.step_size = step_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        #self.name = "q_network"
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DRQN/summaries", sess.graph)


    def network(self):
        #q_net
        scope_var = "q_network"
        with tf.variable_scope(scope_var, reuse=False):
            self.inputs_q = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_q')
            lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='q_lstm' )
            lstm_out, state = tf.nn.dynamic_rnn(lstm, self.inputs_q, dtype=tf.float32)
            reduced_out = lstm_out[:, -1, :]
            reduced_out = tf.reshape(reduced_out, shape=[-1, self.hidden_size])

            w2 = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]))
            b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
            h2 = tf.matmul(reduced_out, w2) + b2
            h2 = tf.nn.relu(h2, 'relu_q')
            h2 = tf.contrib.layers.layer_norm(h2, scope='q_norm')

            out_v = tf.contrib.layers.fully_connected(h2, num_outputs=1, activation_fn=None, scope='q_full_V')
            out_a = tf.contrib.layers.fully_connected(h2, num_outputs=self.action_size, activation_fn=None, scope='q_full_A')
            self.q_value = out_v + (out_a - tf.reduce_mean(out_a, axis=1, keepdims=True))

        #target_net
        scope_tar = "target_network"
        with tf.variable_scope(scope_var, reuse=False):
            self.inputs_target = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_target')
            lstm_t = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='t_lstm')
            lstm_out_t, state_t = tf.nn.dynamic_rnn(lstm_t, self.inputs_target, dtype=tf.float32)
            reduced_out_t = lstm_out_t[:, -1, :]
            reduced_out_t = tf.reshape(reduced_out_t, shape=[-1, self.hidden_size])

            w2_t = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]))
            b2_t = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
            h2_t = tf.matmul(reduced_out_t, w2_t) + b2_t
            h2_t = tf.nn.relu(h2_t, 'relu_t')
            h2_t = tf.contrib.layers.layer_norm(h2_t, scope= 't_norm')

            out_v_t = tf.contrib.layers.fully_connected(h2_t, num_outputs=1, activation_fn=None, scope='t_full_V')
            out_a_t = tf.contrib.layers.fully_connected(h2_t, num_outputs=self.action_size, activation_fn=None, scope='t_full_A')
            self.q_target = out_v_t + (out_a_t - tf.reduce_mean(out_a_t, axis=1, keepdims=True))

        with tf.variable_scope("loss"):
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            with tf.device('/cpu:0'):
                one_hot_actions = tf.one_hot(self.actions, self.action_size)
            Q = tf.reduce_sum(tf.multiply(self.q_value, one_hot_actions), axis=1)
            self.target = tf.placeholder(tf.float32, [None], name='target')
            self.loss = tf.reduce_mean(tf.square(self.target - Q))
            tf.summary.scalar('loss', tf.reduce_mean(self.loss))

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    # def network(self):
    #     with tf.variable_scope(self.name, reuse=False):
    #         self.inputs = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs')
    #         self.actions = tf.placeholder(tf.int32, [None], name='actions')
    #
    #         self.lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
    #         self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm, self.inputs, dtype=tf.float32)
    #         self.reduced_out = self.lstm_out[:, -1, :]
    #         self.reduced_out = tf.reshape(self.reduced_out, shape=[-1, self.hidden_size])
    #
    #         self.w2 = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]))
    #         self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
    #         self.h2 = tf.matmul(self.reduced_out, self.w2) + self.b2
    #         self.h2 = tf.nn.relu(self.h2)
    #         self.h2 = tf.contrib.layers.layer_norm(self.h2)
    #
    #         self.out_v = tf.contrib.layers.fully_connected(self.h2, num_outputs=1, activation_fn=None)
    #         self.out_a = tf.contrib.layers.fully_connected(self.h2, num_outputs=self.action_size, activation_fn=None)
    #         self.out_q = self.out_v + (self.out_a - tf.reduce_mean(self.out_a, axis=1, keepdims=True))
    #
    #
    #         self.target = tf.placeholder(tf.float32, [None], name='target')
    #
    #         with tf.device('/cpu:0'):
    #             one_hot_actions = tf.one_hot(self.actions_, self.action_size)
    #         self.Q = tf.reduce_sum(tf.multiply(self.out_q, one_hot_actions), axis=1)
    #         self.loss = tf.reduce_mean(tf.square(self.target - self.Q))
    #         tf.summary.scalar('loss', tf.reduce_mean(self.loss))
    #
    #         self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, state, reward, action, state_next):
        q_target = self.sess.run(self.q_target, feed_dict={self.inputs_target: state_next})
        #q_target = self.sess.run(self.out_q, feed_dict= {self.inputs: state_next})
        targets = reward + self.gamma * np.max(q_target, axis=1)
        # print(targets)
        # print(np.shape(action))
        # print(action)
        summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                         feed_dict={self.inputs_q: state, self.target: targets,
                                                    self.actions: action})
        return summery, loss

    def chose_action_train(self, current_state):
        #current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        # print(np.shape(current_state))
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        #q = self.sess.run(self.out_q, feed_dict={self.inputs: current_state})
        # print(q)

        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_size)
        else:
            action_chosen = np.argmax(q)
        # print(np.argmax(q))
        return action_chosen

    def chose_action_test(self, current_state):
        #current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        # print(q)
        action_chosen = np.argmax(q)
        # print(np.argmax(q), q)
        return action_chosen

    # upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t, q in zip(target_prmts, q_prmts)])  # ***
        #print("updating target-network parmeters...")

    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, step_size):
        idx = np.random.choice(np.arange(len(self.buffer) - step_size),
                               size=batch_size, replace=False)

        res = []

        for i in idx:
            temp_buffer = []
            for j in range(step_size):
                temp_buffer.append(self.buffer[i + j])
            res.append(temp_buffer)
        return res
