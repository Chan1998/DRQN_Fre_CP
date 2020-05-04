import numpy as np
import random
import sys

class env_network:
    def __init__(self, num_users, num_channels):
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.REWARD = 1

        # self.channel_alloc_freq =
        self.action_space = np.arange(self.NUM_CHANNELS + 1)
        self.users_action = np.zeros([self.NUM_USERS], np.int32)
        self.users_observation = np.zeros([self.NUM_USERS], np.int32)

    def sample(self):
        x = np.random.choice(self.action_space, size=self.NUM_USERS)
        return x

    def step(self, action):
        # print
        assert (action.size) == self.NUM_USERS, "action and user should have same dim {}".format(action)
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1], np.int32)  # 0 for no chnnel access
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0
        for each in action:
            prob = random.uniform(0, 1)
            if prob <= 1:
                self.users_action[j] = each  # action

                channel_alloc_frequency[each] += 1
            j += 1

        for i in range(1, len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0
        # print channel_alloc_frequency
        for i in range(len(action)):

            self.users_observation[i] = channel_alloc_frequency[self.users_action[i]]
            if self.users_action[i] == 0:  # accessing no channel
                self.users_observation[i] = 0
            if self.users_observation[i] == 1:
                reward[i] = 1
            obs.append((self.users_observation[i], reward[i]))
        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1 - residual_channel_capacity
        obs.append(residual_channel_capacity)
        return obs


    def get_states_user(self, batch):
        states = []
        for user in range(self.NUM_USERS):
            states_per_user = []
            for each in batch:
                states_per_batch = []
                for step_i in each:

                    try:
                        states_per_step = step_i[0][user]

                    except IndexError:
                        print(step_i)
                        print("-----------")

                        print("eror")

                        '''for i in batch:
                            print i
                            print "**********"'''
                        sys.exit()
                    states_per_batch.append(states_per_step)
                states_per_user.append(states_per_batch)
            states.append(states_per_user)
        # print len(states)
        return np.array(states)


    def get_actions_user(self, batch):
        actions = []
        for user in range(self.NUM_USERS):
            actions_per_user = []
            for each in batch:
                actions_per_batch = []
                for step_i in each:
                    actions_per_step = step_i[1][user]
                    actions_per_batch.append(actions_per_step)
                actions_per_user.append(actions_per_batch)
            actions.append(actions_per_user)
        return np.array(actions)


    def get_rewards_user(self, batch):
        rewards = []
        for user in range(self.NUM_USERS):
            rewards_per_user = []
            for each in batch:
                rewards_per_batch = []
                for step_i in each:
                    rewards_per_step = step_i[2][user]
                    rewards_per_batch.append(rewards_per_step)
                rewards_per_user.append(rewards_per_batch)
            rewards.append(rewards_per_user)
        return np.array(rewards)


    #
    def get_next_states_user(self, batch):
        next_states = []
        for user in range(self.NUM_USERS):
            next_states_per_user = []
            for each in batch:
                next_states_per_batch = []
                for step_i in each:
                    next_states_per_step = step_i[3][user]
                    next_states_per_batch.append(next_states_per_step)
                next_states_per_user.append(next_states_per_batch)
            next_states.append(next_states_per_user)
        return np.array(next_states)

    def one_hot(self, num, len):
        assert num >= 0 and num < len, "error"
        vec = np.zeros([len], np.int32)
        vec[num] = 1
        return vec

    # generates next-state from action and observation
    def state_generator(self, action, obs):
        input_vector = []
        if action is None:
            print('None')
            sys.exit()
        for user_i in range(action.size):
            input_vector_i = self.one_hot(action[user_i], self.NUM_CHANNELS + 1)
            channel_alloc = obs[-1]
            input_vector_i = np.append(input_vector_i, channel_alloc)
            input_vector_i = np.append(input_vector_i, int(obs[user_i][0]))  # ACK
            input_vector.append(input_vector_i)
        return input_vector
