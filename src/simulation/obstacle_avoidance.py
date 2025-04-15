import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from typing import List, Tuple
import math

class ObstacleAvoidance:
    def __init__(self):
        # Initialize models for each control output
        self.throttle_model = DecisionTreeRegressor(max_depth=3)
        self.brake_model = DecisionTreeRegressor(max_depth=3)
        self.steer_model = LinearRegression()
        
        # Initialize experience buffer
        self.experience_buffer = []
        self.buffer_size = 1000
        
        # Initialize models with basic rules
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with basic rules for speed control."""
        # Create some basic training data
        X = []
        y_throttle = []
        y_brake = []
        y_steer = []
        
        # Case 1: Low speed, straight
        X.append([0.2, 1.0, 0.0, 0.0, 0.0])  # low speed, no obstacles
        y_throttle.append(0.5)  # accelerate
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 2: Medium speed, straight
        X.append([0.5, 1.0, 0.0, 0.0, 0.0])  # medium speed, no obstacles
        y_throttle.append(0.3)  # maintain speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 3: High speed, straight
        X.append([0.8, 1.0, 0.0, 0.0, 0.0])  # high speed, no obstacles
        y_throttle.append(0.1)  # reduce speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 4: Low speed, turn
        X.append([0.2, 1.0, 0.3, 0.0, 0.0])  # low speed, turning
        y_throttle.append(0.3)  # maintain lower speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.3)     # turn
        
        # Case 5: High speed, turn
        X.append([0.8, 1.0, 0.3, 0.0, 0.0])  # high speed, turning
        y_throttle.append(0.0)  # reduce speed
        y_brake.append(0.1)     # light brake
        y_steer.append(0.3)     # turn
        
        # Case 6: Vehicle ahead
        X.append([0.5, 0.2, 0.0, 0.0, 0.0])  # medium speed, vehicle close
        y_throttle.append(0.0)  # no throttle
        y_brake.append(0.3)     # medium brake
        y_steer.append(0.0)     # go straight
        
        # Case 7: Traffic light
        X.append([0.5, 0.1, 0.0, 0.0, 0.0])  # medium speed, near traffic light
        y_throttle.append(0.0)  # no throttle
        y_brake.append(0.5)     # strong brake
        y_steer.append(0.0)     # go straight
        
        # Convert to numpy arrays
        X = np.array(X)
        y_throttle = np.array(y_throttle)
        y_brake = np.array(y_brake)
        y_steer = np.array(y_steer)
        
        # Train models with initial data
        self.throttle_model.fit(X, y_throttle)
        self.brake_model.fit(X, y_brake)
        self.steer_model.fit(X, y_steer)
        
        # Add to experience buffer
        for i in range(len(X)):
            self.experience_buffer.append((X[i], (y_throttle[i], y_brake[i], y_steer[i])))
    
    def preprocess_input(self, 
                        vehicle_location: np.ndarray,
                        vehicle_velocity: float,
                        vehicle_rotation: np.ndarray,
                        obstacles: List[Tuple[np.ndarray, float, str]],
                        next_waypoint: np.ndarray = None) -> np.ndarray:
        """
        Preprocess input data for the models.
        
        Args:
            vehicle_location: Current vehicle location (x, y, z)
            vehicle_velocity: Current vehicle velocity
            vehicle_rotation: Current vehicle rotation (pitch, yaw, roll)
            obstacles: List of tuples containing (location, distance, type) for each obstacle
            next_waypoint: Next waypoint location (x, y, z)
            
        Returns:
            np.ndarray: Preprocessed input features
        """
        # Normalize vehicle velocity (assuming max speed of 50 km/h)
        normalized_velocity = vehicle_velocity / 50.0
        
        # Calculate angle to next waypoint if available
        angle_to_waypoint = 0.0
        if next_waypoint is not None:
            # Calculate vector to waypoint
            waypoint_vector = next_waypoint - vehicle_location
            waypoint_vector = waypoint_vector / np.linalg.norm(waypoint_vector) if np.linalg.norm(waypoint_vector) > 0 else np.zeros(3)
            
            # Calculate vehicle forward vector
            yaw_rad = math.radians(vehicle_rotation[1])  # yaw is the second element
            vehicle_forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0])
            
            # Calculate angle between vehicle forward and waypoint vector
            dot_product = np.dot(vehicle_forward[:2], waypoint_vector[:2])
            angle_to_waypoint = math.acos(np.clip(dot_product, -1.0, 1.0))
            
            # Determine turn direction
            cross_product = np.cross(vehicle_forward[:2], waypoint_vector[:2])
            if cross_product < 0:
                angle_to_waypoint = -angle_to_waypoint
        
        # Get the nearest obstacle
        nearest_obstacle = None
        min_distance = float('inf')
        for obstacle in obstacles:
            if obstacle[1] < min_distance:
                min_distance = obstacle[1]
                nearest_obstacle = obstacle
        
        # Normalize obstacle distance (assuming max detection range of 50 meters)
        normalized_distance = min(min_distance / 50.0, 1.0) if nearest_obstacle else 1.0
        
        # Create input features
        features = np.array([
            normalized_velocity,
            normalized_distance,
            angle_to_waypoint,
            0.0,  # default y position
            0.0   # default z position
        ])
        
        return features
    
    def predict_control(self, 
                       vehicle_location: np.ndarray,
                       vehicle_velocity: float,
                       vehicle_rotation: np.ndarray,
                       obstacles: List[Tuple[np.ndarray, float, str]],
                       next_waypoint: np.ndarray = None) -> Tuple[float, float, float]:
        """
        Predict control actions based on current state.
        
        Args:
            vehicle_location: Current vehicle location (x, y, z)
            vehicle_velocity: Current vehicle velocity
            vehicle_rotation: Current vehicle rotation (pitch, yaw, roll)
            obstacles: List of tuples containing (location, distance, type) for each obstacle
            next_waypoint: Next waypoint location (x, y, z)
            
        Returns:
            Tuple[float, float, float]: Predicted throttle, brake, and steering values
        """
        # Preprocess input
        features = self.preprocess_input(vehicle_location, vehicle_velocity, vehicle_rotation, obstacles, next_waypoint)
        
        # Make predictions
        throttle = self.throttle_model.predict([features])[0]
        brake = self.brake_model.predict([features])[0]
        steer = self.steer_model.predict([features])[0]
        
        # Clip values to valid ranges
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steer = np.clip(steer, -1.0, 1.0)
        
        return throttle, brake, steer
    
    def update_model(self, 
                    vehicle_location: np.ndarray,
                    vehicle_velocity: float,
                    vehicle_rotation: np.ndarray,
                    obstacles: List[Tuple[np.ndarray, float, str]],
                    target_controls: Tuple[float, float, float],
                    next_waypoint: np.ndarray = None):
        """
        Update the models based on new experience.
        
        Args:
            vehicle_location: Current vehicle location (x, y, z)
            vehicle_velocity: Current vehicle velocity
            vehicle_rotation: Current vehicle rotation (pitch, yaw, roll)
            obstacles: List of tuples containing (location, distance, type) for each obstacle
            target_controls: Target control values (throttle, brake, steer)
            next_waypoint: Next waypoint location (x, y, z)
        """
        # Preprocess input
        features = self.preprocess_input(vehicle_location, vehicle_velocity, vehicle_rotation, obstacles, next_waypoint)
        
        # Add experience to buffer
        self.experience_buffer.append((features, target_controls))
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
        
        # Prepare training data
        X = np.array([exp[0] for exp in self.experience_buffer])
        y_throttle = np.array([exp[1][0] for exp in self.experience_buffer])
        y_brake = np.array([exp[1][1] for exp in self.experience_buffer])
        y_steer = np.array([exp[1][2] for exp in self.experience_buffer])
        
        # Update models
        self.throttle_model.fit(X, y_throttle)
        self.brake_model.fit(X, y_brake)
        self.steer_model.fit(X, y_steer) 