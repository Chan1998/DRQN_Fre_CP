import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter

# list = []
# array = np.array(list)
# for i in range (10):
#     i = random.randint(0,1)
#     list.append(i)
#
# array = np.array(list)
# print(list)
# print(array)
#
# pd_data = pd.DataFrame(array)

# print(pd_data)
# pd_data.to_csv("./dataset/pd_data.csv")

# pd_data = pd.read_csv('dataset/DRQN_channl_action.csv')
# #print(np.array(pd_data))
# array = np.array(pd_data)
# array = array[:,-1]
# #print(array)
# plt.scatter(y=np.arange(len(array)),x=array)
# plt.show()
# action_list = []

# for i in range(10):
#     action = [0, 0, 0]
#     for ii in range(2):
#         action[ii] = random.randint(0,2)
#     print(action)
#     action_list.append(action)
#
# array = np.array(action_list)
# print(array)
# pd_data = pd.DataFrame(array,columns=['User1','User2','User3'])
#
# print(pd_data)
# pd_data.to_csv("./dataset/muti_user/pd_data.csv")

# T = np.arctter2(X,Y)
# plt.scatter(X,Y,s=10,c=T,alpha=0.5)
# plt.imshow(a,interpolation='nearest',cmap='bone',origin='upper')
# plt.colorbar(shrink=0.9)
# plt.xticks(())
# plt.yticks(())

#channl_chose_action 散点图

#channel_action:

# pd_data = pd.read_csv('dataset/channel_choose/DRQN_channl_action.csv')
# # #print(np.array(pd_data))
# array = np.array(pd_data)
# DRQN_action = array[:,-1]
# #print(array)
# pd_data = pd.read_csv('dataset/channel_choose/random_channl_action.csv')
# # #print(np.array(pd_data))
# array = np.array(pd_data)
# random_action = array[:,-1]

# plt.figure()
# T= np.arctan2(np.random.normal(0,1,len(DRQN_action)),np.random.normal(0,1,len(DRQN_action)))
# plt.scatter(x=DRQN_action,y=np.arange(len(DRQN_action)),s=30,c=T,alpha=0.5)
# plt.xticks(np.arange(0,16,1))
# plt.xlabel(r"$channel\_choose$")
# plt.ylabel(r"$Time\_slots(s)$")
# plt.yticks(np.arange(0,len(DRQN_action),1000))
# plt.title(r"$DRQN\_action$")
# plt.show()

# plt.figure()
# T= np.arctan2(np.random.normal(0,1,len(random_action)),np.random.normal(0,1,len(random_action)))
# plt.scatter(x=random_action,y=np.arange(len(random_action)),s=30,c=T,alpha=0.5)
# plt.xticks(np.arange(0,16,1))
# plt.xlabel(r"$channel\_choose$")
# plt.ylabel(r"$Time\_slots(s)$")
# plt.yticks(np.arange(0,len(random_action),1000))
# plt.title(r"$random\_action$")
# plt.show()

# num_array = np.zeros(shape=(16))
# #print(num_array)
# #print(DRQN_action)
# arry = Counter(DRQN_action)
# #print(array)
# for i in arry.keys():
#     num_array[i] = float(arry[i])/5184
# #print(num_array)
# DRQN_action_num = num_array
#
# num_array2 = np.zeros(shape=(16))
# #print(num_array)
# #print(DRQN_action)
# arry2 = Counter(random_action)
# #print(array)
# for i in arry2.keys():
#     num_array2[i] = float(arry2[i])/ 5184
# #print(num_array)
# random_action_num = num_array2
#
# #print(DRQN_action_num,random_action_num)
# X = np.arange(len(DRQN_action_num))
# plt.figure()
# plt.xlim(-1,17)
# plt.ylim(-0.3,1)
# plt.xlabel("$channel\_choose$")
# plt.ylabel("$chose\_rate$")
# plt.title("$choose\_action\_probability$")
# plt.bar(X,DRQN_action_num,facecolor='#9999ff',edgecolor='white')
# plt.bar(X,-random_action_num,facecolor='#ff9999',edgecolor='white')
#
# plt.plot([-1,17],[0,0],'r--',lw=1)
#
# for x,y in zip(X,DRQN_action_num):
#     if y >= .01:
#         plt.text(x , y + 0.05, '%.2f'%y,ha='center',va='bottom')
#
# for x,y in zip(X,random_action_num):
#     plt.text(x , -y - 0.05, '%.2f'%y,ha='center',va='top')
#
#
# plt.legend(["$DRQN$","$random$"])
# plt.show()

#channel_reward:
# pd_data = pd.read_csv('dataset/channel_choose/DRQN_channl_reward.csv')
# # #print(np.array(pd_data))
# array = np.array(pd_data)
# DRQN_reward = array[:,-1]
# #print(array)
# pd_data = pd.read_csv('dataset/channel_choose/random_channl_reward.csv')
# # #print(np.array(pd_data))
# array = np.array(pd_data)
# random_reward = array[:,-1]
#
# #print(DRQN_reward.shape,random_reward.shape)
#
# DRQN_reward_2D = DRQN_reward.reshape((72,72))
# random_reward_2D = random_reward.reshape((72,72))
# #print(DRQN_reward_2D.shape)
#
# plt.figure()
# plt.subplot(121)
# plt.imshow(DRQN_reward_2D,interpolation='none',cmap='BuGn',origin='upper')
# cb = plt.colorbar(shrink=0.2)
# cb.set_label("$ACK$")
# cb.set_ticks((0,1))
# plt.xticks(np.arange(0,72,9), rotation=45)
# plt.xlabel(r"$episode$")
# plt.ylabel(r"$Time\_slots(s)$")
# plt.yticks(np.arange(0,72,9), rotation=45)
# plt.title(r"$DRQN\_trans\_ACK$")
#
# plt.subplot(122)
# plt.imshow(random_reward_2D,interpolation='none',cmap='BuGn',origin='upper')
# cb = plt.colorbar(shrink=0.2)
# cb.set_label("$ACK$")
# cb.set_ticks((0,1))
# plt.xticks(np.arange(0,72,9), rotation=45)
# plt.xlabel(r"$episode$")
# plt.ylabel(r"$Time\_slots(s)$")
# plt.yticks(np.arange(0,72,9), rotation=45)
# plt.title(r"$random\_trans\_ACK$")
#
# plt.show()

#channel_state
# pd_data = pd.read_csv('dataset/real_data_trace.csv')
# # # #print(np.array(pd_data))
# array = np.array(pd_data)
# channel_state = array[:5184,-1]
# channel_state_2D = channel_state.reshape((72,72))
#
# plt.figure()
# plt.imshow(channel_state_2D,interpolation='none',cmap='BuGn',origin='upper')
# cb = plt.colorbar(shrink=0.2)
# cb.set_label("$state$")
# cb.set_ticks((0,1))
# plt.xticks(np.arange(0,72,9), rotation=45)
# plt.xlabel(r"$episode$")
# plt.ylabel(r"$Time\_slots(s)$")
# plt.yticks(np.arange(0,72,9), rotation=45)
# plt.title(r"$channel\_state$")
#
# plt.show()

#muti_user_action:
# pd_data = pd.read_csv('dataset/muti_user/DRQN_action.csv')
# #print(np.array(pd_data))
# array = np.array(pd_data)
# DRQN_action1 = array[:,1:]
# DRQN_action1_user = np.zeros((3,100))
# for i in range(3):
#     for ii in range(100):
#         DRQN_action1_user[i][ii] = DRQN_action1[ii][i]
# #print(DRQN_action1_user)
#
# plt.figure()
# plt.imshow(DRQN_action1_user,interpolation='none',cmap='GnBu',origin='upper',aspect='auto')
# cb = plt.colorbar(shrink=0.2)
# cb.set_label("$choose\_channel$")
# cb.set_ticks((0,1,2))
# plt.xticks(np.arange(0,100,10), rotation=45)
# plt.xlabel(r"$Time(s)$")
# plt.ylabel(r"$Users$")
# plt.yticks(np.arange(0,3,1))
# plt.title(r"$muti\_users\_choose\_channel(cooperative)$")
#
# plt.show()
#
# pd_data = pd.read_csv('dataset/muti_user2/DRQN_action.csv')
# #print(np.array(pd_data))
# array = np.array(pd_data)
# DRQN_action1 = array[:,1:]
# DRQN_action1_user = np.zeros((3,100))
# for i in range(3):
#     for ii in range(100):
#         DRQN_action1_user[i][ii] = DRQN_action1[ii][i]
# #print(DRQN_action1_user)
#
# plt.figure()
# plt.imshow(DRQN_action1_user,interpolation='none',cmap='GnBu',origin='upper',aspect='auto')
# cb = plt.colorbar(shrink=0.2)
# cb.set_label("$choose\_channel$")
# cb.set_ticks((0,1,2))
# plt.xticks(np.arange(0,100,10), rotation=45)
# plt.xlabel(r"$Time(s)$")
# plt.ylabel(r"$Users$")
# plt.yticks(np.arange(0,3,1))
# plt.title(r"$muti\_users\_choose\_channel(competitive)$")
#
# plt.show()

pd_data = pd.read_csv('dataset/muti_user/Random_action.csv')
#print(np.array(pd_data))
array = np.array(pd_data)
Random_action1 = array[:,1:]
Random_action1_user = np.zeros((3,100),dtype=int)

Random_action1_user_success = np.zeros((3,100),dtype=int)

for i in range(3):
    for ii in range(100):
        Random_action1_user[i][ii] = Random_action1[ii][i]
#print(DRQN_action1_user)

for i in range(100):
    channel_state = [0,0,0]
    for ii in range(3):
        action = Random_action1_user[ii][i]
        if action:
            channel_state[action]+=1
    for ii in range(3):
        action = Random_action1_user[ii][i]
        if channel_state[action]==1:
            Random_action1_user_success[ii][i] = Random_action1_user[ii][i]



plt.figure()
plt.imshow(Random_action1_user,interpolation='none',cmap='GnBu',origin='upper',aspect='auto')
cb = plt.colorbar(shrink=0.2)
cb.set_label("$choose\_channel$")
cb.set_ticks((0,1,2))
plt.xticks(np.arange(0,100,10), rotation=45)
plt.xlabel(r"$Time(s)$")
plt.ylabel(r"$Users$")
plt.yticks(np.arange(0,3,1))
plt.title(r"$muti\_users\_choose\_channel(Random\ methord)$")

plt.show()

plt.figure()
plt.imshow(Random_action1_user_success,interpolation='none',cmap='GnBu',origin='upper',aspect='auto')
cb = plt.colorbar(shrink=0.2)
cb.set_label("$choose\_channel$")
cb.set_ticks((0,1,2))
plt.xticks(np.arange(0,100,10), rotation=45)
plt.xlabel(r"$Time(s)$")
plt.ylabel(r"$Users$")
plt.yticks(np.arange(0,3,1))
plt.title(r"$muti\_users\_success\_choose\_channel(Random\ methord)$")

plt.show()