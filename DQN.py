import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import time
from skimage import transform
from skimage.color import rgb2gray
import wandb
from collections import deque
import matplotlib.pyplot as plt
from snake import Game
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

wandb.init(
    name="fixed-target",
    project="ai-snake",
    config={
        "learning_rate": 5e-4,
        "decay_rate": 0.0005,
        "gamma": 0.9,
        "pos_reward": 0.1,
        "neg_reward": -1,
        "loss": "mse"
    }
)

config = wandb.config

left = [1,0,0,0]
right = [0,1,0,0]
up = [0,0,1,0]
down = [0,0,0,1]
POSSIBLE_ACTIONS = [left, right, up, down]
NUM_ACTIONS = len(POSSIBLE_ACTIONS)

### MODEL HYPERPARAMETERS
state_size = [84,84,4]
learning_rate =  config.learning_rate

### TRAINING HYPERPARAMETERS
total_epochs = 10000
max_steps = 1000
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0
explore_stop = 0.01
decay_rate = config.decay_rate

# Q learning hyperparameters
gamma = config.gamma

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size
memory_size = 1000000

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    normalized_frame = gray/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame

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

class Agent:
    def __init__(self, env):
        self.env = env
        self.memory = Memory(max_size=memory_size)
        self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(4)], maxlen=4) 

        self.dqn_net = DQNetwork(state_size, learning_rate)
        self.target_net = DQNetwork(state_size, learning_rate)

    def stack_frames(self, state, is_new_episode):
        frame = preprocess_frame(state)
        
        if is_new_episode:
            self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(4)], maxlen=4)
            for _ in range(4):
                self.stacked_frames.append(frame)
        else:
            self.stacked_frames.append(frame)

        return np.stack(self.stacked_frames, axis=2)

    def pre_train(self):
        for i in range(pretrain_length):
            if i == 0:
                state = self.env.reset()
                state = self.stack_frames(state, True)
                
            action = random.choice(POSSIBLE_ACTIONS)
            next_state, reward, terminal = self.env.step(action)
            next_state = self.stack_frames(next_state, False)

            if terminal:
                next_state = np.zeros(state.shape)
                
                self.memory.add((state, action, reward, next_state, terminal))
                
                state = self.env.reset()
                state = self.stack_frames(state, True)                
            else:
                self.memory.add((state, action, reward, next_state, terminal))
                state = next_state

    def train(self):
        decay_step = 0
        all_rewards = []
        
        for epoch in range(total_epochs):
            step = 0
            episode_rewards = []
            
            state = self.env.reset()
            state = self.stack_frames(state, True)
            
            while step < max_steps:
                step += 1            
                decay_step +=1
                
                action, explore_probability = self.dqn_net.predict_action(explore_start, explore_stop, decay_rate, decay_step, state, POSSIBLE_ACTIONS)
                
                next_state, reward, terminal = self.env.step(action)
                episode_rewards.append(reward)

                if terminal:
                    next_state = np.zeros((84,84), dtype=np.int)                    
                    next_state = self.stack_frames(next_state, False)

                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    all_rewards.append(total_reward)

                    print('Epoch: {}'.format(epoch),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {:.4f}'.format(loss))

                    self.memory.add((state, action, reward, next_state, terminal))
                else:
                    next_state = self.stack_frames(next_state, False)                
                    self.memory.add((state, action, reward, next_state, terminal))
                    state = next_state

                batch = self.memory.sample(batch_size)
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

                Qs_state = self.dqn_net.model.predict(states_mb)
                Qs_next_state = self.dqn_net.model.predict(next_states_mb)
                Qs_target_next_state = self.target_net.model.predict(next_states_mb)

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    action = np.argmax(Qs_next_state[i])
                    if terminal:
                        target = rewards_mb[i]                        
                    else:
                        target = rewards_mb[i] + gamma * Qs_target_next_state[i][action]
                    
                    Qs_state[i, actions_mb[i].astype(bool)] = target
                        
                loss = self.dqn_net.model.train_on_batch(states_mb, Qs_state)
            
            self.target_update()

            if epoch % 10 == 0:
                print(np.mean(np.array(all_rewards)[-10:]))
                wandb.log({'Reward': np.mean(np.array(all_rewards)[-10:])})

    def target_update(self):
        weights = self.dqn_net.model.get_weights()
        self.target_net.model.set_weights(weights)

class DQNetwork:
    def __init__(self, state_size, learning_rate):
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
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss=config.loss)
        return model

    def predict_action(self, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            action = random.choice(POSSIBLE_ACTIONS)
            
        else:
            Qs = self.model.predict(state.reshape((1, *state.shape)))
            
            choice = np.argmax(Qs)
            action = POSSIBLE_ACTIONS[int(choice)]
                    
        return action, explore_probability            

def main():
    env = Game(config.pos_reward, config.neg_reward)
    
    agent = Agent(env)
    agent.pre_train()
    agent.train()

if __name__ == "__main__":
    main()