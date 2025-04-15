import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class MLManager:
    def __init__(self):
        """Initialize the ML Manager with neural networks and components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perception_model = self._create_perception_model()
        self.decision_model = self._create_decision_model()
        self.ethical_model = self._create_ethical_model()
        
        # Move models to appropriate device
        self.perception_model.to(self.device)
        self.decision_model.to(self.device)
        self.ethical_model.to(self.device)
        
        # Initialize optimizers
        self.perception_optimizer = torch.optim.Adam(self.perception_model.parameters(), lr=0.001)
        self.decision_optimizer = torch.optim.Adam(self.decision_model.parameters(), lr=0.001)
        self.ethical_optimizer = torch.optim.Adam(self.ethical_model.parameters(), lr=0.001)

    def _create_perception_model(self) -> nn.Module:
        """Create the perception model for processing sensor data."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _create_decision_model(self) -> nn.Module:
        """Create the decision model for making driving decisions."""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # throttle, brake, steer
        )

    def _create_ethical_model(self) -> nn.Module:
        """Create the ethical model for evaluating ethical considerations."""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # ethical priorities: continue, swerve_left, swerve_right
        )

    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process sensor data through the perception model."""
        try:
            # Get camera data and ensure it's in the correct format
            camera_data = sensor_data['camera']
            
            # Convert to tensor and ensure correct shape
            if isinstance(camera_data, np.ndarray):
                # Ensure the array is in the correct format (H, W, C)
                if len(camera_data.shape) == 3:
                    # If channels are first, transpose to (H, W, C)
                    if camera_data.shape[0] == 3:
                        camera_data = np.transpose(camera_data, (1, 2, 0))
                elif len(camera_data.shape) == 2:
                    # If grayscale, convert to RGB
                    camera_data = np.stack([camera_data] * 3, axis=-1)
                
                # Convert to tensor and normalize
                camera_tensor = torch.from_numpy(camera_data).float()
                camera_tensor = camera_tensor / 255.0  # Normalize to [0, 1]
                
                # Add batch dimension and ensure channels are first
                camera_tensor = camera_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Move to device
                camera_tensor = camera_tensor.to(self.device)
                
                # Process through perception model
                with torch.no_grad():
                    classification, regression = self.perception_model(camera_tensor)
                
                return classification, regression
            else:
                raise ValueError("Camera data must be a numpy array")
        except Exception as e:
            print(f"Error processing camera data: {e}")
            # Return zero tensors with correct shape in case of error
            batch_size = 1
            classification_shape = (batch_size, 128)  # Adjust based on your model's output
            regression_shape = (batch_size, 4)  # Adjust based on your model's output
            return (
                torch.zeros(classification_shape, device=self.device),
                torch.zeros(regression_shape, device=self.device)
            )

    def make_decision(self, features: torch.Tensor) -> Dict[str, float]:
        """Make a driving decision based on processed features."""
        with torch.no_grad():
            controls = self.decision_model(features)
            controls = controls.squeeze().cpu().numpy()
        
        return {
            'throttle': float(controls[0]),
            'brake': float(controls[1]),
            'steer': float(controls[2])
        }

    def evaluate_ethics(self, features: torch.Tensor) -> Dict[str, float]:
        """Evaluate ethical considerations based on processed features."""
        with torch.no_grad():
            ethical_scores = self.ethical_model(features)
            ethical_scores = torch.softmax(ethical_scores, dim=1)
            ethical_scores = ethical_scores.squeeze().cpu().numpy()
        
        return {
            'continue': float(ethical_scores[0]),
            'swerve_left': float(ethical_scores[1]),
            'swerve_right': float(ethical_scores[2])
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a training step with the provided batch of data."""
        # Extract data from batch
        features = batch['features'].to(self.device)
        perception_labels = batch['perception_labels'].to(self.device)
        decision_labels = batch['decision_labels'].to(self.device)
        ethical_labels = batch['ethical_labels'].to(self.device)
        
        # Forward pass
        perception_output = self.perception_model(features)
        decision_output = self.decision_model(perception_output)
        ethical_output = self.ethical_model(perception_output)
        
        # Calculate losses
        perception_loss = nn.MSELoss()(perception_output, perception_labels)
        decision_loss = nn.MSELoss()(decision_output, decision_labels)
        ethical_loss = nn.CrossEntropyLoss()(ethical_output, ethical_labels)
        
        # Backward pass
        self.perception_optimizer.zero_grad()
        self.decision_optimizer.zero_grad()
        self.ethical_optimizer.zero_grad()
        
        total_loss = perception_loss + decision_loss + ethical_loss
        total_loss.backward()
        
        # Update weights
        self.perception_optimizer.step()
        self.decision_optimizer.step()
        self.ethical_optimizer.step()
        
        return {
            'perception_loss': float(perception_loss.item()),
            'decision_loss': float(decision_loss.item()),
            'ethical_loss': float(ethical_loss.item()),
            'total_loss': float(total_loss.item())
        }

    def save_models(self, path: str):
        """Save the models to disk."""
        torch.save({
            'perception_model': self.perception_model.state_dict(),
            'decision_model': self.decision_model.state_dict(),
            'ethical_model': self.ethical_model.state_dict(),
            'perception_optimizer': self.perception_optimizer.state_dict(),
            'decision_optimizer': self.decision_optimizer.state_dict(),
            'ethical_optimizer': self.ethical_optimizer.state_dict()
        }, path)

    def load_models(self, path: str):
        """Load the models from disk."""
        checkpoint = torch.load(path)
        self.perception_model.load_state_dict(checkpoint['perception_model'])
        self.decision_model.load_state_dict(checkpoint['decision_model'])
        self.ethical_model.load_state_dict(checkpoint['ethical_model'])
        self.perception_optimizer.load_state_dict(checkpoint['perception_optimizer'])
        self.decision_optimizer.load_state_dict(checkpoint['decision_optimizer'])
        self.ethical_optimizer.load_state_dict(checkpoint['ethical_optimizer']) 