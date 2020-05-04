from  env_model import env_network
from DRQN_agent import DeepQNetwork,Memory
import numpy as np
from collections import  deque
import  matplotlib.pyplot as plt
import tensorflow as tf

TIME_SLOTS = 50000                           # number of time-slots to run simulation
decay_epsilon_STEPS = 500
NUM_CHANNELS = 3                               # Total number of channels
NUM_USERS = 4                                  # Total number of users
# ATTEMPT_PROB = 1                               # attempt probability of ALOHA based  models

memory_size = 10000                      #size of experience replay deque
batch_size = 6                          # Num of batches to train at each time_slot
pretrain_length = batch_size            #this is done to fill the deque up to batch size before training
hidden_size = 128                       #Number of hidden neurons
learning_rate = 0.0001                  #learning rate
# explore_start = .02                     #initial exploration rate
# explore_stop = 0.01                     #final exploration rate
# decay_rate = 0.0001                     #rate of exponential decay of exploration
# gamma = 0.9                             #discount  factor
noise = 0.1
step_size=1+2+2                         #length of history sequence for each datapoint  in batch
state_size = 2 *(NUM_CHANNELS + 1)      #length of input (2 * k + 2)   :k = NUM_CHANNELS
action_size = NUM_CHANNELS+1            #length of output  (k+1)
alpha=0                                 #co-operative fairness constant
beta = 1                                #Annealing constant for Monte - Carlo
interval = 100                           # debug interval
np.random.seed(40)

def reset_env():
    # to sample random actions for each user
    action = env.sample()

    #
    obs = env.step(action)
    state = env.state_generator(action, obs)
    #reward = [i[1] for i in obs[:NUM_USERS]]

    for ii in range(pretrain_length * step_size * 5):
        action = env.sample()
        obs = env.step(
            action)  # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
        next_state = env.state_generator(action, obs)
        reward = [i[1] for i in obs[:NUM_USERS]]
        memory.add((state, action, reward, next_state))
        state = next_state
        history_input.append(state)
    return state, history_input

if __name__ == "__main__":
    # 清空计算图
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        env = env_network(NUM_USERS, NUM_CHANNELS)
        DRQN = DeepQNetwork(learning_rate=learning_rate, state_size=state_size,
                            action_size=action_size, step_size=step_size,
                            hidden_size=hidden_size, sess=sess)
        memory = Memory(max_size=memory_size)
        history_input = deque(maxlen=step_size)  # Input as states
        state, history_input = reset_env()

        loss_list = []
        reward_all_list = []

        ##########################################################################
        ####                      main simulation loop                    ########
        update_iter = 0
        reward_all = 0
        for time_step in range(TIME_SLOTS):
            # initializing action vector
            action = np.zeros([NUM_USERS], dtype=np.int32)
            # converting input history into numpy array
            state_vector = np.array(history_input)
            for each_user in range(NUM_USERS):
                action[each_user] = DRQN.chose_action_train(
                    state_vector[:, each_user].reshape(1, step_size, state_size))

            # taking action as predicted from the q values and receiving the observation from thr envionment
            obs = env.step(action)  # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]

            if time_step % interval == 0:
                print(action)
            #print (obs)

            # Generate next state from action and observation
            next_state = env.state_generator(action, obs)
            # print (next_state)

            # reward for all users given by environment
            reward = [i[1] for i in obs[:NUM_USERS]]

            # calculating sum of rewards
            sum_r = np.sum(reward)
            reward_all += sum_r
            reward_all_list.append(reward_all)
            # calculating cumulative reward
            #cum_r.append(cum_r[-1] + sum_r)

            # If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r)
            #collision = NUM_CHANNELS - sum_r

            # calculating cumulative collision
            #cum_collision.append(cum_collision[-1] + collision)

            #############################
            #  for co-operative policy we will give reward-sum to each user who have contributed
            #  to play co-operatively and rest 0
            # for i in range(len(reward)):
            #     if reward[i] > 0:
            #         reward[i] = sum_r
            #############################


            # total_rewards.append(sum_r)
            # print (reward)


            # add new experiences into the memory buffer as (state, action , reward , next_state) for training
            memory.add((state, action, reward, next_state))

            state = next_state
            # add new experience to generate input-history sequence for next state
            history_input.append(state)

            #  Training block starts
            ###################################################################################

            #  sampling a batch from memory buffer for training
            batch = memory.sample(batch_size, step_size)

            #   matrix of rank 4
            #   shape [NUM_USERS,batch_size,step_size,state_size]
            states = env.get_states_user(batch)

            #   matrix of rank 3
            #   shape [NUM_USERS,batch_size,step_size]
            actions = env.get_actions_user(batch)

            #   matrix of rank 3
            #   shape [NUM_USERS,batch_size,step_size]
            rewards = env.get_rewards_user(batch)

            #   matrix of rank 4
            #   shape [NUM_USERS,batch_size,step_size,state_size]
            next_states = env.get_next_states_user(batch)

            #   Converting [NUM_USERS,batch_size]  ->   [NUM_USERS * batch_size]
            #   first two axis are converted into first axis

            batch_states = np.reshape(states, [-1, states.shape[2], states.shape[3]])
            actions = np.reshape(actions, [-1, actions.shape[2]])
            rewards = np.reshape(rewards, [-1, rewards.shape[2]])
            batch_rewards = rewards[:, -1]
            batch_actions = actions[:, -1]
            batch_next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])



            summery, loss = DRQN.train(state=batch_states,  # 进行训练
                                      reward=batch_rewards,
                                      action=batch_actions,
                                     state_next=batch_next_states
                                      )
            update_iter += 1
            DRQN.write.add_summary(summery, update_iter)
            loss_list.append(loss)
            if time_step % decay_epsilon_STEPS == 0:
                DRQN.decay_epsilon()  # 随训练进行减小探索力度

            # #  creating target vector (possible best action)
            # target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
            #
            # #  Q_target =  reward + gamma * Q_next
            # targets = rewards[:, -1] + gamma * np.max(target_Qs, axis=1)
            #
            # #  calculating loss and train using Adam  optimizer
            # loss, _ = sess.run([mainQN.loss, mainQN.opt],
            #                    feed_dict={mainQN.inputs_: states,
            #                               mainQN.targetQs_: targets,
            #                               mainQN.actions_: actions[:, -1]})


            # print(loss)
            #   Training block ends
            ########################################################################################

            # if  time_step %50000 == 4999:
            #     plt.figure(1)
            #     plt.subplot(311)
            #     #plt.plot(np.arange(1000),total_rewards,"r+")
            #     #plt.xlabel('Time Slots')
            #     #plt.ylabel('total rewards')
            #     #plt.title('total rewards given per time_step')
            #     #plt.show()
            #     plt.plot(np.arange(5001),cum_collision,"r-")
            #     plt.xlabel('Time Slot')
            #     plt.ylabel('cumulative collision')
            #     #plt.show()
            #     plt.subplot(312)
            #     plt.plot(np.arange(5001),cum_r,"r-")
            #     plt.xlabel('Time Slot')
            #     plt.ylabel('Cumulative reward of all users')
            #     #plt.title('Cumulative reward of all users')
            #
            #     plt.subplot(321)
            #     plt.plot(np.arange(len(loss_list)), loss_list, "b-")
            #     plt.xlabel('Time Slot')
            #     plt.ylabel('Loss')
            #
            #
            #     plt.show()
            #
            #     total_rewards = []
            #     cum_r = [0]
            #     cum_collision = [0]
            #     saver.save(sess,'checkpoints/dqn_multi-user.ckpt')
            #     print (time_step,loss , sum(reward) , Qs)

            # print ("*************************************************")
        print(reward_all,TIME_SLOTS*3)
        plt.figure(1)
        plt.subplot(121)
        plt.plot(np.arange(len(loss_list)),loss_list,"r-")
        plt.xlabel('Time Slots')
        plt.ylabel('total loss')
        plt.title('total loss given per time_step')
        plt.subplot(122)
        plt.plot(np.arange(len(reward_all_list)), reward_all_list, "b-")
        plt.xlabel('Time Slots')
        plt.ylabel('total rewards')
        plt.title('total rewards given per time_step')
        plt.show()
        saver = tf.train.Saver()
        saver.save(sess, 'checkpoints/dqn_multi-user.ckpt')





