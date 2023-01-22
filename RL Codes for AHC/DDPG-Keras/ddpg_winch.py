
                                    ### Importing Packages ###

import gym
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import os
#import random

                                ###  Heave Compensation Environment ###
                                
ENV_NAME = 'heave_compensation:heave_compensation-v0'
env = gym.make(ENV_NAME)

                                 ### Action and Observation Space ###

num_states = env.observation_space.shape[0]
print("Size: Observation Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size: Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]          # Upper limit of action space (control input)
lower_bound = env.action_space.low[0]           # Lower limit of action space (control input)

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


                    ### Ornstein-Uhlenbeck process (For generating Action Noise)  ###

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Makes the next noise dependent on the current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

                                            ### Buffer ###

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=128):
        # Number of "experiences" to store at maximum
        self.buffer_capacity = buffer_capacity
        # Number of tuples to train
        self.batch_size = batch_size

        # Indicates the number of times record() was called
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # Have used different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        with tf.GradientTape() as tape:
        # Training and Updating Actor & Critic networks.
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # Computing the Loss and updating the Parameters
    def learn(self):
        # Get Sampling Range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly Sample Indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Converting to Tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# Updates Target Parameters Slowly based on the rate "tau"
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initializing Weights between -3e-3 and 3e-3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out),

    outputs = np.array(outputs)*upper_bound     
    outputs = outputs.tolist()                      
    model = tf.keras.Model(inputs, outputs)
    return model


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



def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding Noise to Action for better exploration
    sampled_actions = sampled_actions.numpy() + noise

    # Making sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

                                    ### Hyper Parameters ###
                                    
std_dev = 0.0005            # Standard Deviation of Action Noise            
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

                            ## Learning Rate for Actor-Critic Models ##

critic_lr = 0.002
actor_lr = 0.001
# critic_lr = 0.001
# actor_lr = 0.0001

                            ## Discount Factor for Future Rewards ##
gamma = 0.998

                                ## For Updating Target Networks ##
tau = 0.001
# tau = 0.01

buffer = Buffer(50000,128)

                                ### Updating the weights ###

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()


# Making the Weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

                                    ### Optimizers ###

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

                                    ### Training ###

ep_reward_list = []
avg_reward_list = []

MAX_EP_STEPS = 3000                         # Number of Episode Time Steps in each Episode
MAX_EPISODES = 150                          # Total Number of Episodes


                ## Initializing some variables for Plotting purpose ##
  
state_zw = np.empty(MAX_EP_STEPS,dtype=np.float32)
state_reference = np.empty(MAX_EP_STEPS,dtype=np.float32)
control_input = np.empty(MAX_EP_STEPS,dtype=np.float32)
error = np.empty(MAX_EP_STEPS,dtype=np.float32)

                ## Loading the Net Heave Time history of the Vessel ##

reference = pd.read_csv('r_moderate.csv')
reference_dot = pd.read_csv('rd_moderate.csv')

reference = np.array(reference.loc[0]) 
reference_dot = np.array(reference_dot.loc[0])

                                    ## Training ##

for ep in range(MAX_EPISODES):

    prev_state, prev_state_complementary = env.reset()
    episodic_reward = 0
   
    for j in range(MAX_EP_STEPS):
           
        
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)

        state, reward, done, info = env.step(action,j,reference,reference_dot)
        
        #control_input[j] = action[0]
        #state_zwdot[j] = state[0]
        #state_zw[j] = state[2]
        #state_reference[j] = state[3]
        #error[j] = state[2] + state[3]
        
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        
        prev_state = state
           
        time_plot = np.arange(0, 300, 0.1);

            
    if done:
        break

    ep_reward_list.append(episodic_reward)

    # Mean of Last 30 Episodes
    avg_reward = np.mean(ep_reward_list[-30:])
    print("Episode * {} * Avg Reward ==> {}* Reward {}".format(ep, avg_reward, episodic_reward))
    avg_reward_list.append(avg_reward)

                    ## Plot of Average Rewards versus Episodes  ##
                    
plt.figure(1)
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Average Episodic Reward")
#axes = plt.gca()
#axes.set_ylim([-500,0])
plt.show()

plt.figure(2)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
#axes = plt.gca()
#axes.set_ylim([-400,0])
plt.show()


                            ## Saving the weights ##

actor_model.save_weights("winch_actor_new.h5")
critic_model.save_weights("winch_critic_new.h5")

target_actor.save_weights("winch_target_actor_new.h5")
target_critic.save_weights("winch_target_critic_new.h5")


