import tensorflow as tf      # Deep Learning library
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
from skimage.color import rgb2gray
from snake import Game

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

left = [1,0,0,0]
right = [0,1,0,0]
up = [0,0,1,0]
down = [0,0,0,1]
POSSIBLE_ACTIONS = [left, right, up, down]
NUM_ACTIONS = len(POSSIBLE_ACTIONS)

# Instansiate env
env = Game()

def preprocess_frame(frame):
    # Greyscale frame 
    gray = rgb2gray(frame)
    
    # Normalize Pixel Values
    normalized_frame = gray/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    
    return preprocessed_frame

stack_size = 4

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
learning_rate =  1e-4      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 10000         # Total episodes for training
max_steps = 1000             # Max possible steps in an episode
batch_size = 64            

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00005           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
#episode_render = False

class DQNetwork:
    def __init__(self, state_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        input = keras.Input(shape=self.state_size)
        x = layers.Conv2D(32, 8, strides=(4,4))(input)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 4, strides=(2,2))(input)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, strides=(2,2))(input)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.Activation('relu')(x)
        output = layers.Dense(NUM_ACTIONS)(x)

        model = keras.Model(input, output)
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

dqn = DQNetwork(state_size, learning_rate)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

# Instantiate memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    action = random.choice(POSSIBLE_ACTIONS)
    next_state, reward, terminal = env.step(action)
    
    #env.render()
    
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    # If the episode is finished (we're dead 3x)
    if terminal:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, terminal))
        
        # Start a new episode
        state = env.reset()
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, terminal))
        
        # Our new state is now the next_state
        state = next_state

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(POSSIBLE_ACTIONS)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = dqn.model.predict(state.reshape((1, *state.shape)))
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = POSSIBLE_ACTIONS[int(choice)]
                
    return action, explore_probability


# TRAINING
if training == True:
    # Initialize the decay rate (that will use to reduce epsilon) 
    decay_step = 0

    all_rewards = []
    
    for episode in range(total_episodes):
        # Set step to 0
        step = 0
        
        # Initialize the rewards of the episode
        episode_rewards = []
        
        # Make a new episode and observe the first state
        state = env.reset()
        
        # Remember that stack frame function also call our preprocess function.
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        while step < max_steps:
            step += 1
            
            #Increase decay_step
            decay_step +=1
            
            # Predict the action to take and take it
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, POSSIBLE_ACTIONS)
            
            #Perform the action and get the next_state, reward, and done information
            next_state, reward, terminal = env.step(action)
            
            # if episode_render:
            #     env.render()
            
            # Add the reward to total reward
            episode_rewards.append(reward)
            
            # If the game is finished
            if terminal:
                # The episode ends so no next state
                next_state = np.zeros((84,84), dtype=np.int)
                
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Set step = max_steps to end the episode
                step = max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)

                all_rewards.append(total_reward)

                print('Episode: {}'.format(episode),
                                'Total reward: {}'.format(total_reward),
                                'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                # Store transition <st,at,rt+1,st+1> in memory D
                memory.add((state, action, reward, next_state, terminal))

            else:
                # Stack the frame of the next_state
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
                # Add experience to memory
                memory.add((state, action, reward, next_state, terminal))

                # st+1 is now our current state
                state = next_state
            
                           
                
            ### LEARNING PART            
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch]) 
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            # Plot stacked_frames
            #f, axarr = plt.subplots(2,2)
            #axarr[0,0].imshow(states_mb[0, :, :, 0])
            #axarr[0,1].imshow(states_mb[0, :, :, 1])
            #axarr[1,0].imshow(states_mb[0, :, :, 2])
            #axarr[1,1].imshow(states_mb[0, :, :, 3])
            #plt.show()

            # Get Q values for next_state
            Qs_state = dqn.model.predict(states_mb)
            Qs_next_state = dqn.model.predict(next_states_mb)
            
            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    #target = rewards_mb[i]
                    target = np.clip(rewards_mb[i], -1, 1)
                    #target_Qs_batch.append(target)
                    
                else:
                    #target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target = np.clip(rewards_mb[i] + gamma * np.max(Qs_next_state[i]), -1, 1)
                
                Qs_state[i, actions_mb[i].astype(bool)] = target
                    
            loss = dqn.model.train_on_batch(states_mb, Qs_state)

        if episode % 10 == 0:
            print(np.mean(np.array(all_rewards)[-10:]))

    plt.plot(all_rewards)
    plt.show()