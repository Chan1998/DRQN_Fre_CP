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
        self.name = "q_network"
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DRQN/summaries", sess.graph)

    def network(self):
        with tf.variable_scope(self.name, reuse=False):
            self.inputs_ = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')

            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm, self.inputs_, dtype=tf.float32)
            self.reduced_out = self.lstm_out[:, -1, :]
            self.reduced_out = tf.reshape(self.reduced_out, shape=[-1, self.hidden_size])

            self.w2 = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
            self.h2 = tf.matmul(self.reduced_out, self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            self.h2 = tf.contrib.layers.layer_norm(self.h2)

            self.out_v = tf.contrib.layers.fully_connected(self.h2, num_outputs=1, activation_fn=None)
            self.out_a = tf.contrib.layers.fully_connected(self.h2, num_outputs=self.action_size, activation_fn=None)
            self.out_q = self.out_v + (self.out_a - tf.reduce_mean(self.out_a, axis=1, keepdims=True))


            self.target = tf.placeholder(tf.float32, [None], name='target')

            with tf.device('/cpu:0'):
                one_hot_actions = tf.one_hot(self.actions_, self.action_size)
            self.Q = tf.reduce_sum(tf.multiply(self.out_q, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.target - self.Q))
            tf.summary.scalar('loss', tf.reduce_mean(self.loss))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, state, reward, action, state_next):
        q_target = self.sess.run(self.out_q, feed_dict={self.inputs_: state_next})
        targets = reward + self.gamma * np.max(q_target, axis=1)
        # print(targets)
        # print(np.shape(action))
        # print(action)
        summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                         feed_dict={self.inputs_: state, self.target: targets,
                                                    self.actions_: action})
        return summery, loss

    def chose_action_train(self, current_state):
        #current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        # print(np.shape(current_state))
        q = self.sess.run(self.out_q, feed_dict={self.inputs_: current_state})
        # print(q)
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_size)
        else:
            action_chosen = np.argmax(q)
        # print(np.argmax(q))
        return action_chosen

    def chose_action_test(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.out_q, feed_dict={self.inputs_: current_state})
        # print(q)
        action_chosen = np.argmax(q)
        # print(np.argmax(q), q)
        return action_chosen

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
