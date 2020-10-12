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
from tqdm import tqdm
from snake_mult_env import Game
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')
from SumTree import SumTree
import pygame
#from google.colab import drive

#drive.mount('/content/gdrive')

wandb.init(
    name="RND div click Mult env Prioritized Dueling Double DQN",
    notes="Two layers, 13 row, split env in 2, static start, 1 divider_prob, 1-8 rnd interval",
    project="ai-snake",
    config={
        "learning_rate": 0.0003,
        "decay_rate": 0.0001,
        "gamma": 0.993,
        "pos_reward": 0.1,
        "neg_reward": -1,
        "explore_stop": 0.001,
        "memory_size": 10000,
    }
)

config = wandb.config

left = [1,0,0,0]
right = [0,1,0,0]
up = [0,0,1,0]
down = [0,0,0,1]
POSSIBLE_ACTIONS = [left, right, up, down]
NUM_ACTIONS = len(POSSIBLE_ACTIONS)

TRAIN_FROM_WEIGHTS = False
USE_RANDOM_CLICKS = True
DIVIDER_PROB = 1
RND_CLICK_PROB = 0.1

### MODEL HYPERPARAMETERS
state_size = [84,84,4]
learning_rate =  config.learning_rate

### TRAINING HYPERPARAMETERS
total_epochs = 100000
max_steps = 1000
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_stop = config.explore_stop
explore_start = explore_stop if TRAIN_FROM_WEIGHTS else 1.0

decay_rate = config.decay_rate

# Q learning hyperparameters
gamma = config.gamma

### MEMORY HYPERPARAMETERS
memory_size = config.memory_size
pretrain_length = memory_size

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    normalized_frame = gray/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame

class Memory(object):
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority
            
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight

            b_idx[i]= index
            
            experience = data
            
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Agent:
    def __init__(self, env):
        self.env = env
        self.memory = Memory(memory_size)
        self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(4)], maxlen=4) 

        self.dqn_net = DQNetwork(state_size, learning_rate)
        self.target_net = DQNetwork(state_size, learning_rate)

        self.dqn_net.model.call = tf.function(self.dqn_net.model.call, experimental_relax_shapes=True)
        self.target_net.model.call = tf.function(self.target_net.model.call, experimental_relax_shapes=True)

        if TRAIN_FROM_WEIGHTS:
            #self.dqn_net.model.load_weights('/content/gdrive/My Drive/vikter/two-layers/model_mean.h5')
            self.dqn_net.model.load_weights('model_single.h5')

    def stack_frames(self, state, is_new_episode):
        frame = preprocess_frame(state)
        
        if is_new_episode:
            self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(4)], maxlen=4)
            for _ in range(4):
                self.stacked_frames.append(frame)
        else:
            self.stacked_frames.append(frame)

        return np.stack(self.stacked_frames, axis=2)

    def play(self):
        self.dqn_net.model.load_weights('model_single.h5')

        for i in range(pretrain_length*10):
            if i == 0:
                state = self.env.reset()
                state = self.stack_frames(state, True)

            Qs = self.dqn_net.model(state.reshape((1, *state.shape)))
            
            choice = np.argmax(Qs)
            action = POSSIBLE_ACTIONS[int(choice)]

            pygame.event.pump()
            mouse_click = None

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_click = pygame.mouse.get_pos()
                    
            next_state, reward, terminal = self.env.step(action, mouse_click)

            next_state = self.stack_frames(next_state, False)

            if terminal:
                next_state = np.zeros(state.shape)
                                
                state = self.env.reset()
                state = self.stack_frames(state, True)                
            else:
                state = next_state
            
    def pre_train(self):
        for i in tqdm(range(pretrain_length)):
            if i == 0:
                state = self.env.reset()
                state = self.stack_frames(state, True)
                
            if TRAIN_FROM_WEIGHTS:
                exp_exp_tradeoff = np.random.rand()
                
                if (explore_stop > exp_exp_tradeoff):
                    action = random.choice(POSSIBLE_ACTIONS)
                else:
                    Qs = self.dqn_net.model(state.reshape((1, *state.shape)))
                
                    choice = np.argmax(Qs)
                    action = POSSIBLE_ACTIONS[int(choice)]
            else:
                action = random.choice(POSSIBLE_ACTIONS)

            if USE_RANDOM_CLICKS and RND_CLICK_PROB > random.random():
                next_state, reward, terminal = self.env.step_with_random_click(action, divider_click_prob = DIVIDER_PROB)
            else: 
                next_state, reward, terminal = self.env.step(action)

            next_state = self.stack_frames(next_state, False)

            if terminal:
                next_state = np.zeros(state.shape)
                
                self.memory.store((state, action, reward, next_state, terminal))
                
                state = self.env.reset()
                state = self.stack_frames(state, True)                
            else:
                self.memory.store((state, action, reward, next_state, terminal))
                state = next_state

    def train(self):
        decay_step = 0
        all_rewards = []
        max_total_single_reward = -999
        max_total_mean_reward = -999
        
        for epoch in range(total_epochs):
            step = 0
            episode_rewards = []
            
            state = self.env.reset()
            state = self.stack_frames(state, True)
            
            while step < max_steps:
                step += 1            
                decay_step +=1
                
                action, explore_probability = self.dqn_net.predict_action(explore_start, explore_stop, decay_rate, decay_step, state, POSSIBLE_ACTIONS)
                
                
                if USE_RANDOM_CLICKS and RND_CLICK_PROB > random.random():
                    next_state, reward, terminal = self.env.step_with_random_click(action, divider_click_prob = DIVIDER_PROB)
                else: 
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

                    self.memory.store((state, action, reward, next_state, terminal))
                else:
                    next_state = self.stack_frames(next_state, False)                
                    self.memory.store((state, action, reward, next_state, terminal))
                    state = next_state

                tree_idx, batch, ISWeights_mb = self.memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                Qs_state = self.dqn_net.model(states_mb).numpy()
                Qs_next_state = self.dqn_net.model(next_states_mb)
                Qs_target_next_state = self.target_net.model(next_states_mb)

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    action = np.argmax(Qs_next_state[i])
                    if terminal:
                        target = rewards_mb[i]
                    else:
                        target = rewards_mb[i] + gamma * Qs_target_next_state[i][action]
                    
                    Qs_state[i, actions_mb[i].astype(bool)] = target
                
                self.dqn_net.is_weights = ISWeights_mb

                y_pred_Q = self.dqn_net.model(states_mb)
                y_pred_Q = y_pred_Q[actions_mb.astype(bool)]

                y_true_Q = Qs_state[actions_mb.astype(bool)]

                absolute_errors = np.abs(y_true_Q - y_pred_Q)

                self.memory.batch_update(tree_idx, absolute_errors)

                loss = self.dqn_net.model.train_on_batch(states_mb, Qs_state)

            self.target_update()

            mean_reward = np.mean(np.array(all_rewards)[-10:])

            if total_reward > max_total_single_reward:
                #self.dqn_net.model.save_weights('/content/gdrive/My Drive/vikter/two-layers/model_single.h5')
                self.dqn_net.model.save_weights('model_single.h5')
                max_total_single_reward = total_reward
                print("Saving weights, maximum single reward reached: ", total_reward)

            if epoch % 10 == 0:
                print(mean_reward)
                wandb.log({'Reward': mean_reward})
                if mean_reward > max_total_mean_reward:
                    #self.dqn_net.model.save_weights('/content/gdrive/My Drive/vikter/two-layers/model_mean.h5')
                    self.dqn_net.model.save_weights('model_mean.h5')
                    max_total_mean_reward = mean_reward
                    print("Saving weights, maximum mean reward reached: ", mean_reward)

    def target_update(self):
        weights = self.dqn_net.model.get_weights()
        self.target_net.model.set_weights(weights)

class DQNetwork:
    def __init__(self, state_size, learning_rate):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.is_weights = None

    def PER_loss(self):
        def loss(y_true, y_pred):   
            return tf.reduce_mean(self.is_weights * tf.math.squared_difference(y_true, y_pred))
        return loss
    
    def build_model(self):
        input = keras.Input(shape=self.state_size)
        x = layers.Conv2D(32, 8, strides=(4,4), activation="relu")(input)
        x = layers.Conv2D(64, 4, strides=(2,2), activation="relu")(x)
        #x = layers.Conv2D(64, 3, strides=(2,2), activation="relu")(x)
        x = layers.Flatten()(x)
        value_fc = layers.Dense(512, activation="relu")(x)
        value = layers.Dense(1)(value_fc)
        advantage_fc = layers.Dense(512, activation="relu")(x)
        advantage = layers.Dense(NUM_ACTIONS)(advantage_fc)
        advantage_norm = layers.Subtract()([advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)])
        aggregation = layers.Add()([value, advantage_norm])
        output = layers.Dense(NUM_ACTIONS)(aggregation)

        model = keras.Model(input, output)
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss=self.PER_loss())
        return model

    def predict_action(self, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            action = random.choice(POSSIBLE_ACTIONS)
            
        else:
            Qs = self.model(state.reshape((1, *state.shape)))
            
            choice = np.argmax(Qs)
            action = POSSIBLE_ACTIONS[int(choice)]
                    
        return action, explore_probability            

def main():
    env = Game(config.pos_reward, config.neg_reward)
    
    agent = Agent(env)
    
    agent.pre_train()
    agent.train()
    #agent.play()

if __name__ == "__main__":
    main()