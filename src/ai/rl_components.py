import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import random

class ReplayBuffer:
    """Buffer to store and sample experience tuples for RL training."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Dict[str, Any]):
        """Store an experience tuple in the buffer."""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class QNetwork(nn.Module):
    """Deep Q-Network for value estimation with pedestrian safety priority."""
    def __init__(self, input_size: int = 256, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        # Main Q-value estimation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 3)  # Output: Q-values for actions
        
        # Pedestrian safety Q-value estimation
        self.pedestrian_fc1 = nn.Linear(input_size, hidden_size)
        self.pedestrian_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.pedestrian_fc3 = nn.Linear(hidden_size // 2, 3)  # Output: Pedestrian safety Q-values
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Main Q-values
        x_main = self.relu(self.fc1(x))
        x_main = self.relu(self.fc2(x_main))
        q_values = self.fc3(x_main)
        
        # Pedestrian safety Q-values
        x_pedestrian = self.relu(self.pedestrian_fc1(x))
        x_pedestrian = self.relu(self.pedestrian_fc2(x_pedestrian))
        pedestrian_q_values = self.pedestrian_fc3(x_pedestrian)
        
        return q_values, pedestrian_q_values

class PolicyNetwork(nn.Module):
    """Policy network for action selection with pedestrian safety priority."""
    def __init__(self, input_size: int = 256, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        # Main policy
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 3)  # Output: action probabilities
        
        # Pedestrian safety policy
        self.pedestrian_fc1 = nn.Linear(input_size, hidden_size)
        self.pedestrian_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.pedestrian_fc3 = nn.Linear(hidden_size // 2, 3)  # Output: pedestrian safety action probabilities
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Main policy
        x_main = self.relu(self.fc1(x))
        x_main = self.relu(self.fc2(x_main))
        action_probs = self.softmax(self.fc3(x_main))
        
        # Pedestrian safety policy
        x_pedestrian = self.relu(self.pedestrian_fc1(x))
        x_pedestrian = self.relu(self.pedestrian_fc2(x_pedestrian))
        pedestrian_probs = self.softmax(self.pedestrian_fc3(x_pedestrian))
        
        return action_probs, pedestrian_probs

class RLManager:
    """Manager class for reinforcement learning components with pedestrian safety priority."""
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """Initialize RL components with enhanced network architecture."""
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Initialize networks with more layers and batch normalization
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        # Initialize optimizers with weight decay for regularization
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=learning_rate,
                                  weight_decay=1e-4)
        
        # Initialize replay buffer with larger capacity
        self.replay_buffer = deque(maxlen=100000)
        
        # Initialize target network
        self.update_target_network()
        
        # Initialize pedestrian safety parameters
        self.min_pedestrian_distance = 5.0
        self.max_pedestrian_speed = 2.0
        self.pedestrian_weight = 0.3
        
        # RL parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001  # Target network update rate
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, torch.Tensor]:
        """Select an action using epsilon-greedy policy with pedestrian safety priority."""
        if training and random.random() < self.epsilon:
            # Random action
            action = random.randrange(self.action_size)
            return action, torch.tensor([action])
        
        # Policy-based action with pedestrian safety priority
        with torch.no_grad():
            action_probs, pedestrian_probs = self.policy_network(state)
            
            # If pedestrians are detected nearby, prioritize pedestrian safety policy
            if 'pedestrian_detected' in state and state['pedestrian_detected']:
                if state['pedestrian_distance'] < self.min_pedestrian_distance:
                    # Use pedestrian safety policy exclusively
                    action = torch.argmax(pedestrian_probs).item()
                    return action, pedestrian_probs
            
            # Blend policies with high weight for pedestrian safety
            blended_probs = (1 - self.pedestrian_weight) * action_probs + \
                          self.pedestrian_weight * pedestrian_probs
            
            action = torch.argmax(blended_probs).item()
            return action, blended_probs
    
    def train_step(self, batch_size: int = 32):
        """Perform one training step using experience replay with pedestrian safety priority."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        # Prepare batch data
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch])
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch])
        pedestrian_scores = torch.tensor([exp['pedestrian_score'] for exp in batch])
        
        # Q-network update
        self.optimizer.zero_grad()
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        # Main Q-learning
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        next_q = next_q_values.max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        q_loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Pedestrian safety Q-learning
        pedestrian_q = current_q_values.gather(1, actions.unsqueeze(1))
        next_pedestrian = next_q_values.max(1)[0].detach()
        target_pedestrian = pedestrian_scores + (1 - dones) * self.gamma * next_pedestrian
        
        pedestrian_q_loss = nn.MSELoss()(pedestrian_q.squeeze(), target_pedestrian)
        
        # Combined loss with high weight for pedestrian safety
        total_q_loss = q_loss + self.pedestrian_weight * pedestrian_q_loss
        total_q_loss.backward()
        self.optimizer.step()
        
        # Policy network update
        action_probs, pedestrian_probs = self.policy_network(states)
        q_values = current_q_values.detach()
        
        # Main policy loss
        policy_loss = -(action_probs * q_values).sum(1).mean()
        
        # Pedestrian safety policy loss
        pedestrian_policy_loss = -(pedestrian_probs * pedestrian_q).sum(1).mean()
        
        # Combined policy loss with high weight for pedestrian safety
        total_policy_loss = policy_loss + self.pedestrian_weight * pedestrian_policy_loss
        total_policy_loss.backward()
        
        # Update target network
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'q_loss': float(q_loss),
            'pedestrian_q_loss': float(pedestrian_q_loss),
            'policy_loss': float(policy_loss),
            'pedestrian_policy_loss': float(pedestrian_policy_loss),
            'epsilon': self.epsilon
        }
    
    def calculate_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """Calculate reward with enhanced steering stability."""
        # Base reward components
        lane_keeping_reward = 0.0
        steering_stability_reward = 0.0
        speed_reward = 0.0
        pedestrian_safety_reward = 0.0
        
        # Lane keeping reward
        lane_offset = state.get('lane_offset', 0.0)
        lane_keeping_reward = -abs(lane_offset) * 0.5  # Penalize deviation from lane center
        
        # Steering stability reward
        current_steer = state.get('steer', 0.0)
        next_steer = next_state.get('steer', 0.0)
        steering_change = abs(next_steer - current_steer)
        steering_stability_reward = -steering_change * 0.3  # Penalize rapid steering changes
        
        # Speed reward
        current_speed = state.get('speed', 0.0)
        target_speed = state.get('target_speed', 3.0)
        speed_diff = abs(current_speed - target_speed)
        speed_reward = -speed_diff * 0.2  # Penalize deviation from target speed
        
        # Pedestrian safety reward
        pedestrian_distance = state.get('pedestrian_distance', float('inf'))
        if pedestrian_distance < self.min_pedestrian_distance:
            pedestrian_safety_reward = -10.0  # Strong penalty for being too close to pedestrians
        elif pedestrian_distance < 15.0:  # Increased safety margin
            pedestrian_safety_reward = -5.0  # Moderate penalty for being near pedestrians
        
        # Combine rewards with weights
        total_reward = (
            lane_keeping_reward * 0.4 +  # Increased weight for lane keeping
            steering_stability_reward * 0.3 +  # Increased weight for steering stability
            speed_reward * 0.2 +
            pedestrian_safety_reward * self.pedestrian_weight
        )
        
        return total_reward
    
    def save_models(self, path: str):
        """Save the trained models to disk."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict()
        }, path)
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.update_target_network()

    def update_target_network(self):
        """Update the target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(param.data) 