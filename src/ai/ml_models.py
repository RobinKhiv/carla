import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.sensor_utils import SensorUtils
from .rl_components import RLManager
from ..ml.ml_manager import MLManager

class PerceptionModel(nn.Module):
    """Enhanced perception model with dedicated object detection layers."""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.detection_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.classification = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.regression = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # x, y, width, height
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv_layers(x)
        detection = self.detection_layers(features)
        classification = self.classification(detection.view(detection.size(0), -1))
        regression = self.regression(detection.view(detection.size(0), -1))
        return classification, regression

class DecisionModel(nn.Module):
    """Enhanced decision model with risk assessment layers."""
    def __init__(self):
        super().__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.control_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # throttle, brake, steer
        )
        
        self.risk_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # risk score
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_processor(x)
        controls = self.control_layers(features)
        risk = self.risk_layers(features)
        return controls, risk

class EthicalModel(nn.Module):
    """Enhanced ethical model with trolley problem handling."""
    def __init__(self):
        super().__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.ethical_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # continue, swerve_left, swerve_right
        )
        
        self.priority_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # ethical priorities
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_processor(x)
        decisions = self.ethical_layers(features)
        priorities = self.priority_layers(features)
        return decisions, priorities

class EnhancedMLManager(MLManager):
    """Enhanced ML Manager that uses the specialized models."""
    def __init__(self):
        super().__init__()
        self.perception_model = PerceptionModel()
        self.decision_model = DecisionModel()
        self.ethical_model = EthicalModel()
        
        # Move models to appropriate device
        self.perception_model.to(self.device)
        self.decision_model.to(self.device)
        self.ethical_model.to(self.device)
        
        # Initialize optimizers
        self.perception_optimizer = torch.optim.Adam(self.perception_model.parameters(), lr=0.001)
        self.decision_optimizer = torch.optim.Adam(self.decision_model.parameters(), lr=0.001)
        self.ethical_optimizer = torch.optim.Adam(self.ethical_model.parameters(), lr=0.001)

    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process sensor data through the enhanced perception model."""
        camera_data = torch.from_numpy(sensor_data['camera']).float().to(self.device)
        
        with torch.no_grad():
            classification, regression = self.perception_model(camera_data)
        
        return classification, regression

    def make_decision(self, features: torch.Tensor) -> Dict[str, float]:
        """Make a driving decision with risk assessment."""
        with torch.no_grad():
            controls, risk = self.decision_model(features)
            controls = controls.squeeze().cpu().numpy()
            risk = risk.squeeze().cpu().numpy()
        
        return {
            'throttle': float(controls[0]),
            'brake': float(controls[1]),
            'steer': float(controls[2]),
            'risk_score': float(risk)
        }

    def evaluate_ethics(self, features: torch.Tensor) -> Dict[str, float]:
        """Evaluate ethical considerations with priorities."""
        with torch.no_grad():
            decisions, priorities = self.ethical_model(features)
            decisions = torch.softmax(decisions, dim=1)
            priorities = torch.softmax(priorities, dim=1)
            decisions = decisions.squeeze().cpu().numpy()
            priorities = priorities.squeeze().cpu().numpy()
        
        return {
            'continue': float(decisions[0]),
            'swerve_left': float(decisions[1]),
            'swerve_right': float(decisions[2]),
            'priorities': {
                'continue': float(priorities[0]),
                'swerve_left': float(priorities[1]),
                'swerve_right': float(priorities[2])
            }
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a training step with enhanced loss functions."""
        features = batch['features'].to(self.device)
        perception_labels = batch['perception_labels'].to(self.device)
        decision_labels = batch['decision_labels'].to(self.device)
        ethical_labels = batch['ethical_labels'].to(self.device)
        
        # Forward pass
        classification, regression = self.perception_model(features)
        controls, risk = self.decision_model(classification)
        decisions, priorities = self.ethical_model(classification)
        
        # Calculate losses
        classification_loss = nn.CrossEntropyLoss()(classification, perception_labels['classification'])
        regression_loss = nn.MSELoss()(regression, perception_labels['regression'])
        control_loss = nn.MSELoss()(controls, decision_labels['controls'])
        risk_loss = nn.MSELoss()(risk, decision_labels['risk'])
        decision_loss = nn.CrossEntropyLoss()(decisions, ethical_labels['decisions'])
        priority_loss = nn.MSELoss()(priorities, ethical_labels['priorities'])
        
        # Backward pass
        self.perception_optimizer.zero_grad()
        self.decision_optimizer.zero_grad()
        self.ethical_optimizer.zero_grad()
        
        total_loss = (classification_loss + regression_loss + 
                     control_loss + risk_loss + 
                     decision_loss + priority_loss)
        total_loss.backward()
        
        # Update weights
        self.perception_optimizer.step()
        self.decision_optimizer.step()
        self.ethical_optimizer.step()
        
        return {
            'classification_loss': float(classification_loss.item()),
            'regression_loss': float(regression_loss.item()),
            'control_loss': float(control_loss.item()),
            'risk_loss': float(risk_loss.item()),
            'decision_loss': float(decision_loss.item()),
            'priority_loss': float(priority_loss.item()),
            'total_loss': float(total_loss.item())
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