
                                    ### Importing Packages ###
                                    
import gym
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy.io
# from mpl_toolkits import mplot3d
#import random

# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'heave_compensation' in env:
#         print("Remove {} from registry".format(env))
#         del gym.envs.registration.registry.env_specs[env]
# import heave_compensation_env

                                ###  Heave Compensation Environment ###
                                
ENV_NAME = 'heave_compensation:heave_compensation-v0'
env = gym.make(ENV_NAME)

                                 ### Action and Observation Space ###

num_states = env.observation_space.shape[0]
print("Size: Observation Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size: Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]          # Upper limit of action space
lower_bound = env.action_space.low[0]           # Lower limit of action space

print("Maximum Value of Action ->  {}".format(upper_bound))
print("Minimum Value of Action ->  {}".format(lower_bound))


MAX_EP_STEPS = 5000                                       
MAX_EPISODES = 1  

                            ## Actor Network ##  

def get_actor():
    # Initializing Weights between -1e-2 and 1-e2
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out),

    outputs = np.array(outputs)*upper_bound     
    outputs = outputs.tolist()         
    model = tf.keras.Model(inputs, outputs)
    return model

                            ## Critic Network ##
                            
def get_critic():
    # States as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(32, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


                            ## Policy ##
def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    # noise = noise_object()
    # Adding Noise to Action for better exploration
    sampled_actions = sampled_actions.numpy()

    # Making sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()


actor_model.load_weights("winch_actor_tuned.h5")
critic_model.load_weights("winch_critic_tuned.h5")

target_actor.load_weights("winch_target_actor_tuned.h5")
target_critic.load_weights("winch_target_critic_tuned.h5") 

                ## Initializing some variables for Plotting purpose ##

state_zwdot = np.empty(MAX_EP_STEPS,dtype=np.float32)   
state_zw = np.empty(MAX_EP_STEPS,dtype=np.float32)
state_reference = np.empty(MAX_EP_STEPS,dtype=np.float32)
control_input = np.empty(MAX_EP_STEPS,dtype=np.float32)
#swash = np.empty(MAX_EP_STEPS,dtype=np.float32)
# reward_plot = np.empty(MAX_EP_STEPS,dtype=np.float32)
# zw_error_plot = np.empty(MAX_EP_STEPS,dtype=np.float32)
# zw_dot_error_plot = np.empty(MAX_EP_STEPS,dtype=np.float32)

                ## Loading the Net Heave Time history of the Vessel ##

# reference = pd.read_csv('r_rl_moderate_offset.csv')
# reference = pd.read_csv('r_rl_moderate_noise_high.csv')
# reference_no_noise = pd.read_csv('r_rl_moderate.csv')
reference = pd.read_csv('r_veryrough.csv')
reference_dot = pd.read_csv('rd_veryrough.csv')
reference = np.array(reference.loc[0]) 
reference_dot = np.array(reference_dot.loc[0])  


                                    ## Testing ##

for ep in range(MAX_EPISODES):
    
    prev_state, prev_state_complementary = env.reset()
    
    for j in range(MAX_EP_STEPS):     
            
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = policy(tf_prev_state)
                state, reward, done, info = env.step(action,j,reference,reference_dot)
                
                control_input[j] = action[0]
                state_zwdot[j] = state[0]
                state_zw[j] = state[2]
                state_reference[j] = state[3]
                #swash[j] = state5[0]

                prev_state = state
                
                time_plot = np.arange(0, 500, 0.1);
                
                
    if done:
        break
    
error = state_zw + state_reference                      # Compensated Motion

# error = state_zw + reference_no_noise[0:100000]       # Compensated Motion
# error = -state_zw - reference_1[0:4000]               # Compensated Motion



plt.figure(1)
sns.set(style="darkgrid")
plt.plot(time_plot,control_input, label='Control Input')
plt.xlabel('Time')
plt.ylabel('Control Input')
axes = plt.gca()
axes.set_ylim([-3,3])
# plt.savefig('up_rough.eps',dpi=1200)     


plt.figure(2)
sns.set(style="darkgrid")
plt.plot(time_plot,state_zw, label='State 4')
plt.xlabel('Time')
plt.ylabel('zw')
plt.show() 

plt.figure(3)
sns.set(style="darkgrid")
plt.plot(time_plot,state_reference, label='State 5')
plt.xlabel('Time')
plt.ylabel('Reference')
plt.show() 

plt.figure(4)
sns.set(style="darkgrid")
plt.plot(time_plot,error, label='Compensated Motion')
plt.xlabel('Time')
plt.ylabel('Compensated Motion')
# axes = plt.gca()
# axes.set_ylim([-0.1,0.1])
# axes.set_xlim([100,300])
# plt.show() 

plt.figure(5)
sns.set(style="darkgrid")
plt.plot(time_plot,error, time_plot,  state_reference, label='Compensated Motion')
plt.legend(['Compensated ', 'Uncompensated'])
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (m)')
# plt.savefig('zw_rough.eps',dpi=1200)   
# axes = plt.gca()
# axes.set_ylim([-1,1.5])



                                    ### Offset Plots ##
                                    
'''
plt.figure(1)
sns.set(style="darkgrid")
plt.plot(time_plot,error, time_plot,  reference_1[0:4000], label='Compensated Motion')
plt.legend(['Compensated net heave at the winch', 'Uncompensated net heave at the winch'])
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (m)')
axes = plt.gca()
axes.set_ylim([-1,1.5])
plt.savefig('offset_moderate.eps',dpi=1200)

plt.figure(2)
sns.set(style="darkgrid")
plt.plot(time_plot,control_input, label='Control Input')
plt.xlabel('Time (sec)')
plt.ylabel('Control Input')
plt.savefig('up_offset_moderate.eps',dpi=1200)
'''


                                        ### 3D Plots ##

'''
zw_error, zw_dot_error = np.meshgrid(zw_error_plot, zw_dot_error_plot)

rewards = 1 - 20*zw_error - 1*zw_dot_error

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(zw_error, zw_dot_error, rewards, 50, cmap='binary')
ax.set_ylabel('zw_dot_error')
ax.set_zlabel('Reward');
plt.savefig('reward_vs_errors.eps',dpi=1200)  

'''




