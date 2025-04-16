import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
import math
import random
import carla

class ObstacleAvoidanceModel(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32, output_size: int = 3):
        super(ObstacleAvoidanceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Output between -1 and 1
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class ObstacleAvoidance:
    def __init__(self):
        self.model = ObstacleAvoidanceModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.experience_buffer = []
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize models with basic rules for speed control."""
        # Create some basic training data
        X = []
        y = []
        
        # Case 1: Low speed, straight
        X.append([0.2, 1.0, 0.0, 0.0, 0.0])  # low speed, no obstacles
        y.append([0.5, 0.0, 0.0])  # throttle, brake, steer
        
        # Case 2: Medium speed, straight
        X.append([0.5, 1.0, 0.0, 0.0, 0.0])  # medium speed, no obstacles
        y.append([0.3, 0.0, 0.0])  # throttle, brake, steer
        
        # Case 3: High speed, straight
        X.append([0.8, 1.0, 0.0, 0.0, 0.0])  # high speed, no obstacles
        y.append([0.1, 0.0, 0.0])  # throttle, brake, steer
        
        # Case 4: Pedestrian directly ahead - emergency stop and steer
        X.append([0.5, 0.1, 0.0, 0.0, 1.0])  # medium speed, pedestrian very close
        y.append([0.0, 0.8, -0.8])  # full brake, strong steer left
        
        # Case 5: Pedestrian directly ahead - emergency stop and steer
        X.append([0.5, 0.1, 0.0, 0.0, 1.0])  # medium speed, pedestrian very close
        y.append([0.0, 0.8, 0.8])  # full brake, strong steer right
        
        # Case 6: Pedestrian slightly to the left - steer right
        X.append([0.5, 0.2, -0.2, 0.0, 1.0])  # medium speed, pedestrian left
        y.append([0.2, 0.4, 0.7])  # moderate brake, strong steer right
        
        # Case 7: Pedestrian slightly to the right - steer left
        X.append([0.5, 0.2, 0.2, 0.0, 1.0])  # medium speed, pedestrian right
        y.append([0.2, 0.4, -0.7])  # moderate brake, strong steer left
        
        # Case 8: Multiple pedestrians ahead - emergency maneuver
        X.append([0.6, 0.15, 0.0, 0.0, 1.0])  # medium-high speed, multiple pedestrians
        y.append([0.0, 0.9, -0.9])  # full brake, maximum steer left
        
        # Case 9: Multiple pedestrians ahead - emergency maneuver
        X.append([0.6, 0.15, 0.0, 0.0, 1.0])  # medium-high speed, multiple pedestrians
        y.append([0.0, 0.9, 0.9])  # full brake, maximum steer right
        
        # Case 10: High speed with pedestrian ahead - emergency stop
        X.append([0.8, 0.1, 0.0, 0.0, 1.0])  # high speed, pedestrian very close
        y.append([0.0, 1.0, 0.0])  # full brake, no steer (emergency stop)
        
        # Case 11: Pedestrian crossing from left - steer right
        X.append([0.5, 0.2, -0.3, 0.0, 1.0])  # medium speed, pedestrian crossing left
        y.append([0.1, 0.5, 0.8])  # strong brake, maximum steer right
        
        # Case 12: Pedestrian crossing from right - steer left
        X.append([0.5, 0.2, 0.3, 0.0, 1.0])  # medium speed, pedestrian crossing right
        y.append([0.1, 0.5, -0.8])  # strong brake, maximum steer left
        
        # Case 13: Pedestrian cluster in middle - slow and steer
        X.append([0.4, 0.15, 0.0, 0.0, 1.0])  # medium speed, pedestrian cluster
        y.append([0.0, 0.7, -0.7])  # strong brake, strong steer left
        
        # Case 14: Pedestrian cluster in middle - slow and steer
        X.append([0.4, 0.15, 0.0, 0.0, 1.0])  # medium speed, pedestrian cluster
        y.append([0.0, 0.7, 0.7])  # strong brake, strong steer right
        
        # Case 15: Complex scenario with vehicles and pedestrians
        X.append([0.6, 0.2, 0.3, 0.3, 1.0])  # medium-high speed, mixed obstacles
        y.append([0.0, 0.8, -0.8])  # full brake, maximum steer left
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Train the model
        num_epochs = 1000
        print("\nTraining obstacle avoidance model...")
        for epoch in range(num_epochs):
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Progress: [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        print("Obstacle avoidance model training complete")
        
        # Add to experience buffer
        for i in range(len(X)):
            self.experience_buffer.append((X[i].numpy(), y[i].numpy()))
    
    def preprocess_input(self, vehicle_location: np.ndarray, vehicle_velocity: float, 
                        vehicle_rotation: np.ndarray, obstacles: List[Tuple[np.ndarray, float, str]], 
                        next_waypoint_location: np.ndarray) -> np.ndarray:
        """Preprocess input data for the model."""
        # Calculate normalized speed (0 to 1)
        normalized_speed = min(float(vehicle_velocity) / 50.0, 1.0)  # Assuming max speed of 50 km/h
        
        # Calculate distance to nearest obstacle
        min_distance = 1.0  # Default to max distance
        for _, distance, _ in obstacles:
            min_distance = min(min_distance, float(distance) / 50.0)  # Normalize by max distance of 50m
        
        # Calculate angle to next waypoint
        vehicle_forward = np.array([
            math.cos(math.radians(float(vehicle_rotation[1]))),
            math.sin(math.radians(float(vehicle_rotation[1]))),
            0
        ])
        
        # Handle Vector3D objects by accessing their components
        if hasattr(next_waypoint_location, 'x'):
            direction = np.array([
                float(next_waypoint_location.x) - float(vehicle_location[0]),
                float(next_waypoint_location.y) - float(vehicle_location[1]),
                float(next_waypoint_location.z) - float(vehicle_location[2])
            ])
        else:
            direction = np.array([
                float(next_waypoint_location[0]) - float(vehicle_location[0]),
                float(next_waypoint_location[1]) - float(vehicle_location[1]),
                float(next_waypoint_location[2]) - float(vehicle_location[2])
            ])
            
        direction = direction / np.linalg.norm(direction)
        angle = math.degrees(math.acos(np.dot(vehicle_forward, direction)))
        normalized_angle = min(abs(angle) / 90.0, 1.0)  # Normalize by max angle of 90 degrees
        
        # Check for traffic light state
        traffic_light_state = 0.0  # Default to no traffic light
        for _, _, obstacle_type in obstacles:
            if obstacle_type == 'traffic_light':
                traffic_light_state = 1.0
                break
        
        # Check for pedestrian state
        pedestrian_state = 0.0  # Default to no pedestrian
        for _, _, obstacle_type in obstacles:
            if obstacle_type == 'pedestrian':
                pedestrian_state = 1.0
                break
        
        return np.array([
            normalized_speed,
            min_distance,
            normalized_angle,
            traffic_light_state,
            pedestrian_state
        ])
    
    def predict_control(self, vehicle_location: np.ndarray, vehicle_velocity: float, 
                       vehicle_rotation: np.ndarray, obstacles: List[Tuple[np.ndarray, float, str]], 
                       next_waypoint_location: np.ndarray) -> Tuple[float, float, float]:
        """Predict control values using the model."""
        # Preprocess input
        input_data = self.preprocess_input(
            vehicle_location, vehicle_velocity, vehicle_rotation, 
            obstacles, next_waypoint_location
        )
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Convert output to Python floats
        throttle = float((output[0].item() + 1) / 2)  # Convert from [-1, 1] to [0, 1]
        brake = float((output[1].item() + 1) / 2)     # Convert from [-1, 1] to [0, 1]
        steer = float(output[2].item())                # Already in [-1, 1]
        
        # Clip values to ensure they're within valid ranges
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        steer = max(-1.0, min(1.0, steer))
        
        return throttle, brake, steer
    
    def evaluate_space(self, space_location: np.ndarray, vehicle_speed: float, 
                      obstacles: List[Tuple[np.ndarray, float, str]]) -> bool:
        """Evaluate if a space is safe to navigate through using ML model."""
        try:
            # Preprocess input for space evaluation
            input_data = self.preprocess_input(
                space_location,  # Use space location instead of vehicle location
                vehicle_speed,
                np.array([0, 0, 0]),  # Simplified rotation
                obstacles,
                space_location  # Use same location for waypoint
            )
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_data)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Extract safety score from model output
            safety_score = float(output[0].item())  # First output neuron for safety
            
            # Consider space safe if safety score is above threshold
            return safety_score > 0.5
            
        except Exception as e:
            print(f"Error evaluating space: {e}")
            return False
    
    def score_space(self, space_location: np.ndarray, pedestrian_location: np.ndarray, 
                   vehicle_speed: float) -> float:
        """Score a space based on its desirability for navigation."""
        try:
            # Calculate distance to pedestrian
            distance_to_pedestrian = np.linalg.norm(space_location - pedestrian_location)
            
            # Normalize distance (assuming max distance of 50m)
            normalized_distance = min(distance_to_pedestrian / 50.0, 1.0)
            
            # Calculate speed factor (slower is better when near pedestrians)
            speed_factor = 1.0 - min(vehicle_speed / 50.0, 1.0)  # Assuming max speed of 50 km/h
            
            # Combine factors with weights
            score = (0.7 * normalized_distance + 0.3 * speed_factor)
            
            return float(score)
            
        except Exception as e:
            print(f"Error scoring space: {e}")
            return 0.0 