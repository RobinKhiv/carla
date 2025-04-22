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
    def __init__(self, world=None):
        self.model = ObstacleAvoidanceModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.experience_buffer = []
        self.world = world  # Store the world reference
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
        
        # Case 4: Pedestrian during right turn - early detection
        X.append([0.5, 0.3, 0.4, 0.0, 1.0])  # medium speed, turning right, pedestrian ahead
        y.append([0.1, 0.6, -0.7])  # reduce speed, brake, steer left to avoid
        
        # Case 5: Pedestrian during left turn - early detection
        X.append([0.5, 0.3, -0.4, 0.0, 1.0])  # medium speed, turning left, pedestrian ahead
        y.append([0.1, 0.6, 0.7])  # reduce speed, brake, steer right to avoid
        
        # Case 6: Pedestrian in curve - gradual avoidance
        X.append([0.4, 0.4, 0.3, 0.0, 1.0])  # medium speed, in curve, pedestrian ahead
        y.append([0.2, 0.5, -0.6])  # moderate speed, brake, gradual steer left
        
        # Case 7: Pedestrian in curve - gradual avoidance
        X.append([0.4, 0.4, -0.3, 0.0, 1.0])  # medium speed, in curve, pedestrian ahead
        y.append([0.2, 0.5, 0.6])  # moderate speed, brake, gradual steer right
        
        # Case 8: Pedestrian during sharp turn - emergency maneuver
        X.append([0.3, 0.2, 0.6, 0.0, 1.0])  # medium speed, sharp turn, pedestrian close
        y.append([0.0, 0.8, -0.8])  # full brake, maximum steer left
        
        # Case 9: Pedestrian during sharp turn - emergency maneuver
        X.append([0.3, 0.2, -0.6, 0.0, 1.0])  # medium speed, sharp turn, pedestrian close
        y.append([0.0, 0.8, 0.8])  # full brake, maximum steer right
        
        # Case 10: Multiple pedestrians in curve - early detection
        X.append([0.4, 0.5, 0.3, 0.0, 1.0])  # medium speed, in curve, multiple pedestrians
        y.append([0.1, 0.7, -0.7])  # reduce speed, strong brake, strong steer left
        
        # Case 11: Multiple pedestrians in curve - early detection
        X.append([0.4, 0.5, -0.3, 0.0, 1.0])  # medium speed, in curve, multiple pedestrians
        y.append([0.1, 0.7, 0.7])  # reduce speed, strong brake, strong steer right
        
        # Case 12: High speed in curve with pedestrian - emergency stop
        X.append([0.7, 0.2, 0.4, 0.0, 1.0])  # high speed, in curve, pedestrian close
        y.append([0.0, 1.0, 0.0])  # full brake, no steer (emergency stop)
        
        # Case 13: Pedestrian crossing during turn - early reaction
        X.append([0.4, 0.4, 0.2, 0.0, 1.0])  # medium speed, turning, pedestrian crossing
        y.append([0.1, 0.6, -0.6])  # reduce speed, brake, steer left
        
        # Case 14: Pedestrian crossing during turn - early reaction
        X.append([0.4, 0.4, -0.2, 0.0, 1.0])  # medium speed, turning, pedestrian crossing
        y.append([0.1, 0.6, 0.6])  # reduce speed, brake, steer right
        
        # Case 15: Complex scenario with vehicles and pedestrians in curve
        X.append([0.5, 0.3, 0.3, 0.3, 1.0])  # medium speed, in curve, mixed obstacles
        y.append([0.0, 0.8, -0.8])  # full brake, maximum steer left
        
        # New cases for lane changes and complex scenarios
        # Case 16: Lane change into pedestrian - abort maneuver
        X.append([0.5, 0.2, 0.4, 0.0, 1.0])  # medium speed, changing lanes, pedestrian in target lane
        y.append([0.0, 0.7, -0.7])  # full brake, steer back to original lane
        
        # Case 17: Lane change into pedestrian - abort maneuver
        X.append([0.5, 0.2, -0.4, 0.0, 1.0])  # medium speed, changing lanes, pedestrian in target lane
        y.append([0.0, 0.7, 0.7])  # full brake, steer back to original lane
        
        # Case 18: Multiple lane changes with pedestrians
        X.append([0.6, 0.3, 0.5, 0.0, 1.0])  # medium-high speed, multiple lane changes, pedestrian ahead
        y.append([0.1, 0.6, -0.6])  # reduce speed, brake, steer to safe lane
        
        # Case 19: Multiple lane changes with pedestrians
        X.append([0.6, 0.3, -0.5, 0.0, 1.0])  # medium-high speed, multiple lane changes, pedestrian ahead
        y.append([0.1, 0.6, 0.6])  # reduce speed, brake, steer to safe lane
        
        # Case 20: Pedestrian between lanes - find safe path
        X.append([0.4, 0.3, 0.0, 0.0, 1.0])  # medium speed, pedestrian between lanes
        y.append([0.2, 0.5, -0.5])  # moderate speed, brake, steer to clear side
        
        # Case 21: Pedestrian between lanes - find safe path
        X.append([0.4, 0.3, 0.0, 0.0, 1.0])  # medium speed, pedestrian between lanes
        y.append([0.2, 0.5, 0.5])  # moderate speed, brake, steer to clear side
        
        # Case 22: High speed lane change with pedestrian - emergency maneuver
        X.append([0.7, 0.2, 0.4, 0.0, 1.0])  # high speed, changing lanes, pedestrian in path
        y.append([0.0, 0.9, -0.9])  # full brake, maximum steer to avoid
        
        # Case 23: High speed lane change with pedestrian - emergency maneuver
        X.append([0.7, 0.2, -0.4, 0.0, 1.0])  # high speed, changing lanes, pedestrian in path
        y.append([0.0, 0.9, 0.9])  # full brake, maximum steer to avoid
        
        # Case 24: Complex scenario with multiple lane changes and pedestrians
        X.append([0.5, 0.3, 0.3, 0.3, 1.0])  # medium speed, multiple lane changes, mixed obstacles
        y.append([0.1, 0.7, -0.7])  # reduce speed, strong brake, strong steer to safe path
        
        # Case 25: Complex scenario with multiple lane changes and pedestrians
        X.append([0.5, 0.3, -0.3, 0.3, 1.0])  # medium speed, multiple lane changes, mixed obstacles
        y.append([0.1, 0.7, 0.7])  # reduce speed, strong brake, strong steer to safe path
        
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
        
        # Get current waypoint and check lane availability
        try:
            current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
            if current_waypoint:
                # Check if we're in the leftmost lane
                left_lane = current_waypoint.get_left_lane()
                right_lane = current_waypoint.get_right_lane()
                
                # If we're in the leftmost lane and trying to steer left, reverse the steering
                if left_lane is None and steer < 0:
                    print("[ML] Reversing steering direction: No left lane available")
                    steer = -steer  # Reverse the steering direction
                
                # If we're in the rightmost lane and trying to steer right, reverse the steering
                if right_lane is None and steer > 0:
                    print("[ML] Reversing steering direction: No right lane available")
                    steer = -steer  # Reverse the steering direction
                
                # If we have both lanes available, choose the one with fewer obstacles
                if left_lane and right_lane:
                    left_obstacles = sum(1 for _, dist, _ in obstacles if dist < 20.0 and 
                                       self._is_obstacle_on_left(vehicle_location, vehicle_rotation, _))
                    right_obstacles = sum(1 for _, dist, _ in obstacles if dist < 20.0 and 
                                        self._is_obstacle_on_right(vehicle_location, vehicle_rotation, _))
                    
                    if left_obstacles < right_obstacles and steer > 0:
                        print("[ML] Choosing left lane: Fewer obstacles")
                        steer = -abs(steer)  # Steer left
                    elif right_obstacles < left_obstacles and steer < 0:
                        print("[ML] Choosing right lane: Fewer obstacles")
                        steer = abs(steer)  # Steer right
        except Exception as e:
            print(f"Error checking lane availability: {e}")
        
        # Clip values to ensure they're within valid ranges
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        steer = max(-1.0, min(1.0, steer))
        
        return throttle, brake, steer
    
    def _is_obstacle_on_left(self, vehicle_location: np.ndarray, vehicle_rotation: np.ndarray, 
                            obstacle_location: np.ndarray) -> bool:
        """Check if an obstacle is on the left side of the vehicle."""
        vehicle_forward = np.array([
            math.cos(math.radians(vehicle_rotation[1])),
            math.sin(math.radians(vehicle_rotation[1])),
            0
        ])
        vehicle_right = np.array([-vehicle_forward[1], vehicle_forward[0], 0])
        
        obstacle_direction = obstacle_location - vehicle_location
        obstacle_direction = obstacle_direction / np.linalg.norm(obstacle_direction)
        
        return np.dot(obstacle_direction, vehicle_right) > 0
    
    def _is_obstacle_on_right(self, vehicle_location: np.ndarray, vehicle_rotation: np.ndarray, 
                             obstacle_location: np.ndarray) -> bool:
        """Check if an obstacle is on the right side of the vehicle."""
        vehicle_forward = np.array([
            math.cos(math.radians(vehicle_rotation[1])),
            math.sin(math.radians(vehicle_rotation[1])),
            0
        ])
        vehicle_right = np.array([-vehicle_forward[1], vehicle_forward[0], 0])
        
        obstacle_direction = obstacle_location - vehicle_location
        obstacle_direction = obstacle_direction / np.linalg.norm(obstacle_direction)
        
        return np.dot(obstacle_direction, vehicle_right) < 0
    
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