import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
from typing import List, Tuple, Dict, Any
import carla

class EthicalPriorities:
    def __init__(self, pedestrian_weight: float = 1.0, passenger_weight: float = 1.0, 
                 property_weight: float = 0.5, traffic_law_weight: float = 0.8):
        self.pedestrian_weight = pedestrian_weight
        self.passenger_weight = passenger_weight
        self.property_weight = property_weight
        self.traffic_law_weight = traffic_law_weight

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RLEthicalAgent:
    def __init__(self, state_size: int, action_size: int, ethical_priorities: EthicalPriorities):
        self.state_size = state_size
        self.action_size = action_size
        self.ethical_priorities = ethical_priorities
        
        # DQN parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = ReplayBuffer(10000)
        
        # Initialize networks
        self.policy_net = DQN(state_size, 64, action_size)
        self.target_net = DQN(state_size, 64, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def get_state(self, vehicle, world) -> np.ndarray:
        """Get the current state of the environment."""
        try:
            # Vehicle state
            vehicle_location = np.array([
                vehicle.get_location().x,
                vehicle.get_location().y,
                vehicle.get_location().z
            ])
            vehicle_velocity = vehicle.get_velocity()
            speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6
            
            # Get nearby actors
            nearby_actors = []
            for actor in world.get_actors():
                if actor.id != vehicle.id:
                    distance = actor.get_location().distance(vehicle.get_location())
                    if distance < 50.0:  # Only consider actors within 50 meters
                        actor_type = self._get_actor_type(actor)
                        nearby_actors.append((
                            np.array([
                                actor.get_location().x,
                                actor.get_location().y,
                                actor.get_location().z
                            ]),
                            distance,
                            actor_type
                        ))
            
            # Get traffic light state
            traffic_light_state = self._get_traffic_light_state(vehicle, world)
            
            # Get next waypoint
            current_waypoint = world.get_map().get_waypoint(vehicle.get_location())
            next_waypoint = current_waypoint.next(1.0)[0] if current_waypoint else None
            
            # Combine all state information
            state = np.zeros(self.state_size)
            state[0] = speed / 50.0  # Normalized speed
            state[1] = len(nearby_actors) / 10.0  # Normalized number of nearby actors
            state[2] = traffic_light_state
            if next_waypoint:
                state[3] = next_waypoint.transform.location.x
                state[4] = next_waypoint.transform.location.y
            
            return state
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size)
    
    def _get_actor_type(self, actor) -> str:
        """Get the type of an actor."""
        if 'vehicle' in actor.type_id:
            return 'vehicle'
        elif 'walker' in actor.type_id:
            return 'pedestrian'
        elif 'traffic.traffic_light' in actor.type_id:
            return 'traffic_light'
        return 'other'
    
    def _get_traffic_light_state(self, vehicle, world) -> float:
        """Get the state of the nearest traffic light."""
        for traffic_light in world.get_actors().filter('traffic.traffic_light'):
            if traffic_light.get_location().distance(vehicle.get_location()) < 30.0:
                state = traffic_light.get_state()
                if state == carla.TrafficLightState.Red:
                    return 0.0
                elif state == carla.TrafficLightState.Yellow:
                    return 0.5
                elif state == carla.TrafficLightState.Green:
                    return 1.0
        return 1.0  # No traffic light nearby
    
    def _calculate_ethical_reward(self, vehicle, world) -> float:
        """Calculate reward based on ethical considerations."""
        try:
            reward = 0.0
            
            # Check for pedestrians
            for actor in world.get_actors().filter('walker.*'):
                distance = actor.get_location().distance(vehicle.get_location())
                if distance < 10.0:  # Close to pedestrian
                    reward -= self.ethical_priorities.pedestrian_weight * (10.0 - distance) / 10.0
            
            # Check for traffic violations
            if self._is_violating_traffic_laws(vehicle, world):
                reward -= self.ethical_priorities.traffic_law_weight
            
            # Check for property damage risk
            if self._is_risking_property_damage(vehicle, world):
                reward -= self.ethical_priorities.property_weight
            
            return float(reward)
        except Exception as e:
            print(f"Error calculating ethical reward: {e}")
            return 0.0
    
    def _is_violating_traffic_laws(self, vehicle, world) -> bool:
        """Check if the vehicle is violating traffic laws."""
        # Check speed limit
        speed = vehicle.get_velocity().length() * 3.6  # Convert to km/h
        if speed > 50.0:  # Speed limit
            return True
        
        # Check traffic lights
        for traffic_light in world.get_actors().filter('traffic.traffic_light'):
            if traffic_light.get_location().distance(vehicle.get_location()) < 10.0:
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    return True
        
        return False
    
    def _is_risking_property_damage(self, vehicle, world) -> bool:
        """Check if the vehicle is risking property damage."""
        # Check distance to static objects
        for actor in world.get_actors().filter('static.*'):
            if actor.get_location().distance(vehicle.get_location()) < 5.0:
                return True
        return False
    
    def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def train(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        try:
            # Compute Q(s_t, a)
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Compute Q(s_{t+1}, a)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss and update
            loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.item()
        except Exception as e:
            print(f"Error in RL training: {e}")
            return None
    
    def update_target_network(self):
        """Update the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.push(state, action, reward, next_state, done) 