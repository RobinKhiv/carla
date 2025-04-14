from typing import Dict, Any
from .ml_models import MLManager

class DecisionMaker:
    def __init__(self):
        """Initialize the decision maker with ML models."""
        self.ml_manager = MLManager()
        self.last_decision = None
        
    def make_decision(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Make a driving decision based on sensor data using ML models.
        
        Args:
            sensor_data: Dictionary containing sensor readings
            
        Returns:
            Dictionary containing control commands
        """
        try:
            # Process sensor data through ML models
            processed_features = self.ml_manager.process_sensor_data(sensor_data)
            
            # Get control commands from decision model
            controls = self.ml_manager.make_decision(processed_features)
            
            # Get ethical considerations
            ethical_weights = self.ml_manager.evaluate_ethics(processed_features)
            
            # Store decision for training
            self.last_decision = {
                'controls': controls,
                'ethical_weights': ethical_weights,
                'sensor_data': sensor_data
            }
            
            return controls
            
        except Exception as e:
            print(f"Error in decision making: {e}")
            # Return safe default controls
            return {
                'throttle': 0.0,
                'brake': 1.0,
                'steer': 0.0
            }
    
    def train_on_experience(self, experience_data: Dict[str, Any]):
        """
        Train the ML models on collected experience data.
        
        Args:
            experience_data: Dictionary containing training data
        """
        try:
            # Perform training step
            losses = self.ml_manager.train_step(experience_data)
            
            # Print training progress
            print(f"Training losses - Perception: {losses['perception_loss']:.4f}, "
                  f"Decision: {losses['decision_loss']:.4f}, "
                  f"Ethical: {losses['ethical_loss']:.4f}")
            
        except Exception as e:
            print(f"Error in training: {e}")
    
    def save_models(self, path: str):
        """Save the trained ML models to disk."""
        self.ml_manager.save_models(path)
    
    def load_models(self, path: str):
        """Load trained ML models from disk."""
        self.ml_manager.load_models(path) 