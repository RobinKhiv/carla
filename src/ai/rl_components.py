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
    def __init__(self, state_size: int = 256, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize networks
        self.q_network = QNetwork(state_size)
        self.target_q_network = QNetwork(state_size)
        self.policy_network = PolicyNetwork(state_size)
        
        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters())
        self.policy_optimizer = optim.Adam(self.policy_network.parameters())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # RL parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001  # Target network update rate
        
        # Pedestrian safety parameters
        self.pedestrian_weight = 0.8  # High weight for pedestrian safety
        self.min_pedestrian_distance = 5.0  # Minimum safe distance to pedestrians
        self.max_pedestrian_speed = 1.5  # Maximum speed when pedestrians are nearby
        
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
        self.q_optimizer.zero_grad()
        current_q_values, current_pedestrian_q = self.q_network(states)
        next_q_values, next_pedestrian_q = self.target_q_network(next_states)
        
        # Main Q-learning
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        next_q = next_q_values.max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        q_loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Pedestrian safety Q-learning
        current_pedestrian = current_pedestrian_q.gather(1, actions.unsqueeze(1))
        next_pedestrian = next_pedestrian_q.max(1)[0].detach()
        target_pedestrian = pedestrian_scores + (1 - dones) * self.gamma * next_pedestrian
        
        pedestrian_q_loss = nn.MSELoss()(current_pedestrian.squeeze(), target_pedestrian)
        
        # Combined loss with high weight for pedestrian safety
        total_q_loss = q_loss + self.pedestrian_weight * pedestrian_q_loss
        total_q_loss.backward()
        self.q_optimizer.step()
        
        # Policy network update
        self.policy_optimizer.zero_grad()
        action_probs, pedestrian_probs = self.policy_network(states)
        q_values, pedestrian_q = self.q_network(states).detach()
        
        # Main policy loss
        policy_loss = -(action_probs * q_values).sum(1).mean()
        
        # Pedestrian safety policy loss
        pedestrian_policy_loss = -(pedestrian_probs * pedestrian_q).sum(1).mean()
        
        # Combined policy loss with high weight for pedestrian safety
        total_policy_loss = policy_loss + self.pedestrian_weight * pedestrian_policy_loss
        total_policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target network
        for target_param, param in zip(self.target_q_network.parameters(), 
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
    
    def calculate_reward(self, state: Dict[str, Any], action: int, 
                        next_state: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate reward based on state transition with pedestrian safety priority."""
        reward = 0.0
        pedestrian_score = 0.0
        
        # Pedestrian safety rewards (highest priority)
        if 'pedestrian_detected' in state and state['pedestrian_detected']:
            distance = state['pedestrian_distance']
            if distance < self.min_pedestrian_distance:
                # Severe penalty for being too close to pedestrians
                reward -= 20.0
                pedestrian_score -= 1.0
            elif distance < 2 * self.min_pedestrian_distance:
                # Moderate penalty for being in warning zone
                reward -= 10.0
                pedestrian_score -= 0.5
            else:
                # Reward for maintaining safe distance
                reward += 5.0
                pedestrian_score += 0.5
                
            # Speed control near pedestrians
            if state['speed'] > self.max_pedestrian_speed:
                reward -= 5.0
                pedestrian_score -= 0.3
            else:
                reward += 2.0
                pedestrian_score += 0.2
        
        # Safety rewards (secondary priority)
        if state['risk_score'] > 0.8:
            reward -= 10.0
        elif state['risk_score'] < 0.2:
            reward += 1.0
            
        # Progress rewards (lowest priority)
        if next_state['distance_traveled'] > state['distance_traveled']:
            reward += 0.1
            
        # Comfort rewards (lowest priority)
        if abs(action) < 0.3:  # Smooth actions
            reward += 0.05
            
        return reward, min(max(pedestrian_score, 0.0), 1.0)
    
    def save_models(self, path: str):
        """Save the trained models to disk."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict()
        }, path)
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network']) 