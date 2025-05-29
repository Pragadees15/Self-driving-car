import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import datetime
import copy
import json
import os
from collections import deque
from matplotlib import pyplot

import sys
sys.path.append(".")
from Utils import global_settings as gs

class DQNModel(nn.Module):
    """
    PyTorch implementation of a Deep Q-Network with optional batch normalization
    """
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        # Check if batch normalization should be used
        self.use_batch_norm = gs.Q_LEARNING_SETTINGS.get("USE_BATCH_NORM", True)
        
        # Enhanced network architecture
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_dim)
        
        # Batch normalization layers (if enabled)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(128)
            self.bn5 = nn.BatchNorm1d(64)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Handle single sample input during inference with batch norm
        if self.use_batch_norm and x.dim() == 2 and x.size(0) == 1 and self.training:
            # For single sample in training mode, switch to non-batch-norm forward path
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            return self.fc6(x)
        elif self.use_batch_norm:
            # Use batch normalization
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = F.relu(self.bn4(self.fc4(x)))
            x = F.relu(self.bn5(self.fc5(x)))
            return self.fc6(x)
        else:
            # No batch normalization (more stable for single samples)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            return self.fc6(x)

class ReplayBuffer:
    """
    Experience replay buffer to store and sample experiences
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)

class PyTorchDQN:
    """
    PyTorch implementation of Deep Q-Learning with experience replay and target networks
    """
    def __init__(self, state_n, actions_n, load=None):
        self.state_n = state_n
        self.actions_n = actions_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Print GPU information for verification
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
        
        # Parameters from global settings
        settings = gs.Q_LEARNING_SETTINGS
        self.learning_rate = settings["LEARNING_RATE"]
        self.discount_rate = settings["DISCOUNT_RATE"]
        self.epsilon = settings["EPSILON_PROBABILITY"] if settings["TRAINING"] else 0
        self.epsilon_decay = settings["EPSILON_DECAY"]
        self.min_epsilon = settings["EPSILON_MIN"]
        self.network_copy_steps = settings["TARGET_NET_COPY_STEPS"]
        self.max_buffer_length = settings["BUFFER_LENGTH"]
        self.batch_size = settings["BATCH_SIZE"]  # Get batch size from global settings
        
        # Initialize policy and target networks
        self.policy_net = DQNModel(state_n, actions_n).to(self.device)
        self.target_net = DQNModel(state_n, actions_n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer with momentum
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate, 
            betas=(settings["GD_MOMENTUM"], 0.999)
        )
        
        # Learning rate scheduler for adaptive learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Initialize experience replay buffer
        self.experience_buffer = ReplayBuffer(self.max_buffer_length)
        
        # Load model if specified
        if load is not None:
            self.load_model(load)
            print("LOADED: ", load)
            
        # Statistics
        self.frame_num = 0
        self.reward_cache = []
        self.error_cache = []
        
    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon and gs.Q_LEARNING_SETTINGS["TRAINING"]:
            action = random.randint(0, self.actions_n-1)
            q_values = [0] * self.actions_n
            q_values[action] = 1
            return action, q_values
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor([state]).to(self.device)
        
        # Make sure the policy network is in eval mode for inference
        was_training = self.policy_net.training
        self.policy_net.eval()
        
        # Get q-values from policy network
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Restore previous training state if needed
        if was_training:
            self.policy_net.train()
            
        # Select action with highest q-value
        return np.argmax(q_values), q_values.tolist()
    
    def update_experience_buffer(self, state, action, reward, next_state=None, done=False):
        # If next_state is not provided, assume it's the same as current state (for backward compatibility)
        if next_state is None:
            next_state = state
            
        self.experience_buffer.push(state, action, reward, next_state, done)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_exploration_probability(self):
        self.epsilon = max(self.epsilon * np.exp(-self.epsilon_decay), self.min_epsilon)
    
    def train(self, verbose=1):
        # Skip if not enough experiences
        if len(self.experience_buffer) < self.batch_size:
            return
            
        total_loss = 0
        train_iterations = int(len(self.experience_buffer) * gs.Q_LEARNING_SETTINGS["TRAIN_AMOUNT"] / self.batch_size)
        
        # Set policy network to training mode
        self.policy_net.train()
        
        for _ in range(train_iterations):
            # Sample from replay buffer
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.experience_buffer.sample(self.batch_size)
            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            reward_batch = reward_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)
            done_batch = done_batch.to(self.device)
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
            
            # Compute V(s_{t+1}) for all next states
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1)[0]
            
            # Compute the expected Q values: r + gamma * max_a' Q(s', a')
            expected_state_action_values = reward_batch + (1 - done_batch) * self.discount_rate * next_state_values
            
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients (normalization)
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update target network
            self.frame_num += 1
            if self.frame_num % self.network_copy_steps == 0:
                self.update_target_network()
        
        # Store average loss
        if train_iterations > 0:
            avg_loss = total_loss / train_iterations
            self.error_cache.append(avg_loss)
            
        # Store episode reward
        total_reward = sum(item[2] for item in self.experience_buffer.buffer)
        self.reward_cache.append(total_reward)
        
        # Update learning rate based on reward
        if len(self.reward_cache) % 10 == 0:
            self.scheduler.step(total_reward)
        
        # Clear buffer after training
        self.experience_buffer = ReplayBuffer(self.max_buffer_length)
        
        # Set policy network back to evaluation mode
        self.policy_net.eval()
        
        if verbose >= 2:
            print("Exploration:", self.epsilon)
            print("Avg Loss:", avg_loss if train_iterations > 0 else "N/A")
            print("Reward:", total_reward)
            print("Learning Rate:", self.optimizer.param_groups[0]['lr'])
            if self.device.type == 'cuda':
                print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            print("============================================\n")
    
    @property
    def mean_rewards(self):
        if not self.reward_cache:
            return 0
        return int(sum(self.reward_cache) / len(self.reward_cache))
    
    def save_model(self, type):
        name = gs.SAVED_MODELS_ROOT + type + "_model_" + datetime.datetime.now().strftime("%d.%m;%H.%M") + "_" + str(self.mean_rewards)
        
        # Create directory if it doesn't exist
        os.makedirs(gs.SAVED_MODELS_ROOT, exist_ok=True)
        
        # Save model state
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'frame_num': self.frame_num,
            'reward_cache': self.reward_cache,
            'error_cache': self.error_cache,
            'state_n': self.state_n,
            'actions_n': self.actions_n
        }, name + '.pt')
        
        # Save settings
        with open(name + "_settings.txt", "w") as file:
            file.write(json.dumps(gs.Q_LEARNING_SETTINGS, separators=(", ", ":")).replace(" ", "\n"))
            
        print(f"Model saved to {name}.pt")
    
    def load_model(self, path):
        full_path = gs.SAVED_MODELS_ROOT + path + '.pt'
        
        if not os.path.exists(full_path):
            print(f"Model file not found: {full_path}")
            return
            
        # Load checkpoint to CPU first, then transfer to device
        checkpoint = torch.load(full_path, map_location='cpu')
        
        # Check if state dimensions match
        loaded_state_n = checkpoint.get('state_n', self.state_n)
        loaded_actions_n = checkpoint.get('actions_n', self.actions_n)
        
        # Reinitialize networks if dimensions don't match
        if loaded_state_n != self.state_n or loaded_actions_n != self.actions_n:
            print(f"Warning: Model dimensions don't match (loaded: {loaded_state_n}x{loaded_actions_n}, current: {self.state_n}x{self.actions_n})")
            print("Reinitializing networks with loaded weights...")
            self.policy_net = DQNModel(loaded_state_n, loaded_actions_n).to(self.device)
            self.target_net = DQNModel(loaded_state_n, loaded_actions_n).to(self.device)
            self.state_n = loaded_state_n
            self.actions_n = loaded_actions_n
        
        # Load model weights
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Handle scheduler state if it exists
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Move models to proper device
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        
        # Fix optimizer state to work on correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.frame_num = checkpoint.get('frame_num', 0)
        self.reward_cache = checkpoint.get('reward_cache', [])
        self.error_cache = checkpoint.get('error_cache', [])
        
        # Update target network
        self.target_net.eval()
        
        print(f"Model loaded from {full_path}")
        print(f"State dimensions: {self.state_n}x{self.actions_n}")
        print(f"Current epsilon: {self.epsilon}")
        if self.reward_cache:
            print(f"Mean reward: {sum(self.reward_cache) / len(self.reward_cache):.2f}")
        
    def reward_graph(self, **kwargs):
        """
        Plot the reward history
        """
        print("LEARNING RATE:", self.learning_rate)
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(self.reward_cache, **kwargs)
        pyplot.title('Rewards per Episode')
        pyplot.xlabel('Episode')
        pyplot.ylabel('Total Reward')
        pyplot.grid(True)
        pyplot.show()
        
    def error_graph(self, **kwargs):
        """
        Plot the error/loss history
        """
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(self.error_cache, **kwargs)
        pyplot.title('Loss per Episode')
        pyplot.xlabel('Episode')
        pyplot.ylabel('Loss')
        pyplot.grid(True)
        pyplot.show() 