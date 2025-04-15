import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.sensor_utils import SensorUtils
from .rl_components import RLManager

class PerceptionModel(nn.Module):
    """Neural network for processing sensor data and detecting objects."""
    def __init__(self, input_size: int = 1024, hidden_size: int = 512):
        super(PerceptionModel, self).__init__()
        # Input processing layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 256)  # Output: object features
        
        # Object detection layers
        self.detection_fc1 = nn.Linear(256, 128)
        self.detection_fc2 = nn.Linear(128, 64)
        self.detection_fc3 = nn.Linear(64, 32)  # Output: object detection features
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process input features
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        base_features = self.fc3(x)
        
        # Process object detection
        detection = self.relu(self.detection_fc1(base_features))
        detection = self.relu(self.detection_fc2(detection))
        detection = self.detection_fc3(detection)
        
        return base_features, detection

class DecisionModel(nn.Module):
    """Neural network for making driving decisions."""
    def __init__(self, input_size: int = 256):
        super(DecisionModel, self).__init__()
        self.input_size = input_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Control output layers
        self.control_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Changed from 2 to 3 for throttle, brake, and steer
        )
        
        # Risk assessment layers
        self.risk_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # single risk score
            nn.Sigmoid()  # ensure output is between 0 and 1
        )
    
    def forward(self, x):
        # Ensure input is a tensor and has the correct shape
        if isinstance(x, tuple):
            x = x[0]  # Take the first element if input is a tuple
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Verify input size
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.shape[1]}")
        
        # Process through feature extractor
        features = self.feature_extractor(x)
        
        # Get control outputs
        controls = self.control_layers(features)
        
        # Get risk score
        risk_score = self.risk_layers(features)
        
        return controls, risk_score

class EthicalModel(nn.Module):
    """Neural network for ethical decision making."""
    def __init__(self, input_size: int = 256, hidden_size: int = 128):
        super(EthicalModel, self).__init__()
        # Main ethical layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 5)  # Output: ethical priorities
        
        # Trolley problem layers
        self.trolley_fc1 = nn.Linear(input_size, hidden_size)
        self.trolley_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.trolley_fc3 = nn.Linear(hidden_size // 2, 3)  # Output: trolley decision
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process main ethical priorities
        ethics = self.relu(self.fc1(x))
        ethics = self.relu(self.fc2(ethics))
        priorities = self.softmax(self.fc3(ethics))
        
        # Process trolley problem
        trolley = self.relu(self.trolley_fc1(x))
        trolley = self.relu(self.trolley_fc2(trolley))
        trolley_decision = self.softmax(self.trolley_fc3(trolley))
        
        return priorities, trolley_decision

class MLManager:
    """Manager class for ML models in the autonomous vehicle system."""
    def __init__(self):
        self.perception_model = PerceptionModel()
        self.decision_model = DecisionModel()
        self.ethical_model = EthicalModel()
        self.rl_manager = RLManager()
        self.sensor_utils = SensorUtils()
        
        # Initialize optimizers
        self.perception_optimizer = optim.Adam(self.perception_model.parameters())
        self.decision_optimizer = optim.Adam(self.decision_model.parameters())
        self.ethical_optimizer = optim.Adam(self.ethical_model.parameters())
        
        # Loss functions
        self.perception_loss = nn.MSELoss()
        self.decision_loss = nn.MSELoss()
        self.ethical_loss = nn.KLDivLoss()
        self.trolley_loss = nn.CrossEntropyLoss()
        
        # Training state
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process raw sensor data through the perception model."""
        # Convert sensor data to tensor
        features = self.sensor_utils.extract_features(sensor_data)
        features_tensor = torch.FloatTensor(features)
        
        # Process through perception model
        with torch.no_grad():
            base_features, detection_features = self.perception_model(features_tensor)
        
        return base_features, detection_features
    
    def make_decision(self, processed_features: torch.Tensor, 
                     training: bool = True) -> Dict[str, Any]:
        """Make driving decisions based on processed features."""
        with torch.no_grad():
            # Ensure features are in the correct format
            if isinstance(processed_features, tuple):
                processed_features = processed_features[0]  # Take the first element if input is a tuple
            if not isinstance(processed_features, torch.Tensor):
                processed_features = torch.tensor(processed_features, dtype=torch.float32)
            if len(processed_features.shape) == 1:
                processed_features = processed_features.unsqueeze(0)  # Add batch dimension if missing
            
            # Get base controls and risk score
            controls, risk_score = self.decision_model(processed_features)
            
            # Get ethical considerations
            priorities, trolley_decision = self.ethical_model(processed_features)
            
            # Combine with RL if training
            if training:
                rl_action, action_probs = self.rl_manager.select_action(
                    processed_features, training=True)
                
                # Blend RL action with base controls
                controls = self._blend_controls(controls, rl_action)
            
            # Convert tensors to Python scalars
            controls = controls.squeeze().cpu().numpy()
            risk_score = risk_score.squeeze().item()
            priorities = priorities.squeeze().cpu().numpy()
            trolley_decision = trolley_decision.squeeze().cpu().numpy()
            
            # Convert to control dictionary
            decision = {
                'controls': {
                    'throttle': float(controls[0]),
                    'brake': float(controls[1]),
                    'steer': float(controls[2])
                },
                'risk_score': float(risk_score),
                'ethical_weights': {
                    'pedestrian_safety': float(priorities[0]),
                    'passenger_safety': float(priorities[1]),
                    'other_vehicle_safety': float(priorities[2]),
                    'property_damage': float(priorities[3]),
                    'traffic_rules': float(priorities[4])
                },
                'trolley_decision': {
                    'continue_straight': float(trolley_decision[0]),
                    'swerve_left': float(trolley_decision[1]),
                    'swerve_right': float(trolley_decision[2])
                }
            }
            
            return decision
    
    def _blend_controls(self, base_controls: torch.Tensor, 
                       rl_action: int) -> torch.Tensor:
        """Blend base controls with RL action."""
        # Define action mappings
        action_mappings = {
            0: torch.tensor([0.7, 0.0, 0.0]),  # Accelerate
            1: torch.tensor([0.0, 0.7, 0.0]),  # Brake
            2: torch.tensor([0.0, 0.0, 0.5])   # Steer
        }
        
        # Get RL control
        rl_control = action_mappings[rl_action]
        
        # Blend with base controls
        alpha = 0.3  # RL influence factor
        blended_controls = (1 - alpha) * base_controls + alpha * rl_control
        
        return blended_controls
    
    def update_rl(self, state: Dict[str, Any], action: int, 
                 next_state: Dict[str, Any], done: bool):
        """Update RL components with new experience."""
        # Calculate reward
        reward = self.rl_manager.calculate_reward(state, action, next_state)
        
        # Store experience
        experience = {
            'state': torch.FloatTensor(state['features']),
            'action': action,
            'reward': reward,
            'next_state': torch.FloatTensor(next_state['features']),
            'done': done
        }
        self.rl_manager.replay_buffer.push(experience)
        
        # Train RL if enough experiences
        if len(self.rl_manager.replay_buffer) >= 32:
            rl_losses = self.rl_manager.train_step()
            if rl_losses:
                print(f"RL Training - Q Loss: {rl_losses['q_loss']:.4f}, "
                      f"Policy Loss: {rl_losses['policy_loss']:.4f}, "
                      f"Epsilon: {rl_losses['epsilon']:.4f}")
    
    def train_step(self, batch_data: Dict[str, Any]):
        """Perform one training step on the models."""
        # Extract features and labels
        features = torch.FloatTensor(batch_data['features'])
        perception_labels = torch.FloatTensor(batch_data['perception_labels'])
        decision_labels = torch.FloatTensor(batch_data['decision_labels'])
        ethical_labels = torch.FloatTensor(batch_data['ethical_labels'])
        trolley_labels = torch.LongTensor(batch_data['trolley_labels'])
        
        # Train perception model
        self.perception_optimizer.zero_grad()
        base_features, detection_features = self.perception_model(features)
        perception_loss = self.perception_loss(detection_features, perception_labels)
        perception_loss.backward()
        self.perception_optimizer.step()
        
        # Train decision model
        self.decision_optimizer.zero_grad()
        controls, risk_score = self.decision_model(features)
        decision_loss = self.decision_loss(controls, decision_labels)
        decision_loss.backward()
        self.decision_optimizer.step()
        
        # Train ethical model
        self.ethical_optimizer.zero_grad()
        priorities, trolley_decision = self.ethical_model(features)
        ethical_loss = self.ethical_loss(priorities.log(), ethical_labels)
        trolley_loss = self.trolley_loss(trolley_decision, trolley_labels)
        total_ethical_loss = ethical_loss + trolley_loss
        total_ethical_loss.backward()
        self.ethical_optimizer.step()
        
        return {
            'perception_loss': float(perception_loss),
            'decision_loss': float(decision_loss),
            'ethical_loss': float(ethical_loss),
            'trolley_loss': float(trolley_loss)
        }
    
    def save_models(self, path: str):
        """Save the trained models to disk."""
        torch.save({
            'perception_model': self.perception_model.state_dict(),
            'decision_model': self.decision_model.state_dict(),
            'ethical_model': self.ethical_model.state_dict(),
            'rl_models': {
                'q_network': self.rl_manager.q_network.state_dict(),
                'policy_network': self.rl_manager.policy_network.state_dict(),
                'target_q_network': self.rl_manager.target_q_network.state_dict()
            }
        }, path)
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        checkpoint = torch.load(path)
        self.perception_model.load_state_dict(checkpoint['perception_model'])
        self.decision_model.load_state_dict(checkpoint['decision_model'])
        self.ethical_model.load_state_dict(checkpoint['ethical_model'])
        
        # Load RL models if present
        if 'rl_models' in checkpoint:
            self.rl_manager.q_network.load_state_dict(checkpoint['rl_models']['q_network'])
            self.rl_manager.policy_network.load_state_dict(checkpoint['rl_models']['policy_network'])
            self.rl_manager.target_q_network.load_state_dict(checkpoint['rl_models']['target_q_network'])

    def evaluate_ethics(self, processed_features: torch.Tensor) -> Dict[str, Any]:
        """Evaluate ethical considerations based on processed features."""
        with torch.no_grad():
            # Ensure features are in the correct format
            if isinstance(processed_features, tuple):
                processed_features = processed_features[0]
            if not isinstance(processed_features, torch.Tensor):
                processed_features = torch.tensor(processed_features, dtype=torch.float32)
            if len(processed_features.shape) == 1:
                processed_features = processed_features.unsqueeze(0)
            
            # Get ethical considerations
            priorities, trolley_decision = self.ethical_model(processed_features)
            
            # Convert tensors to Python scalars
            priorities = priorities.squeeze().cpu().numpy()
            trolley_decision = trolley_decision.squeeze().cpu().numpy()
            
            # Convert to ethical evaluation dictionary
            ethics = {
                'ethical_weights': {
                    'pedestrian_safety': float(priorities[0]),
                    'passenger_safety': float(priorities[1]),
                    'other_vehicle_safety': float(priorities[2]),
                    'property_damage': float(priorities[3]),
                    'traffic_rules': float(priorities[4])
                },
                'trolley_decision': {
                    'continue_straight': float(trolley_decision[0]),
                    'swerve_left': float(trolley_decision[1]),
                    'swerve_right': float(trolley_decision[2])
                }
            }
            
            return ethics 