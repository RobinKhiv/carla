import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from typing import List, Tuple

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
        """Initialize models with basic rules for obstacle avoidance."""
        # Create some basic training data
        X = []
        y_throttle = []
        y_brake = []
        y_steer = []
        
        # Case 1: No obstacles, high speed
        X.append([0.8, 1.0, 0.0, 0.0, 0.0])  # high speed, far obstacle
        y_throttle.append(0.7)  # maintain speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 2: Obstacle far ahead
        X.append([0.8, 0.5, 0.0, 1.0, 0.0])  # high speed, medium distance
        y_throttle.append(0.5)  # reduce speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 3: Obstacle close ahead
        X.append([0.8, 0.2, 0.0, 1.0, 0.0])  # high speed, close distance
        y_throttle.append(0.0)  # no throttle
        y_brake.append(0.5)     # medium brake
        y_steer.append(0.0)     # go straight
        
        # Case 4: Obstacle to the left
        X.append([0.8, 0.3, -1.0, 0.0, 0.0])  # high speed, obstacle left
        y_throttle.append(0.3)  # reduce speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.3)     # steer right
        
        # Case 5: Obstacle to the right
        X.append([0.8, 0.3, 1.0, 0.0, 0.0])  # high speed, obstacle right
        y_throttle.append(0.3)  # reduce speed
        y_brake.append(0.0)     # no brake
        y_steer.append(-0.3)    # steer left
        
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
                        obstacles: List[Tuple[np.ndarray, float, str]]) -> np.ndarray:
        """
        Preprocess input data for the models.
        
        Args:
            vehicle_location: Current vehicle location (x, y, z)
            vehicle_velocity: Current vehicle velocity
            vehicle_rotation: Current vehicle rotation (pitch, yaw, roll)
            obstacles: List of tuples containing (location, distance, type) for each obstacle
            
        Returns:
            np.ndarray: Preprocessed input features
        """
        # Normalize vehicle velocity (assuming max speed of 50 km/h)
        normalized_velocity = vehicle_velocity / 50.0
        
        # Get the nearest obstacle in front of the vehicle
        nearest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in obstacles:
            location, distance, _ = obstacle
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle
        
        if nearest_obstacle is None:
            # If no obstacles, use default values
            obstacle_location = np.array([0, 0, 0])
            obstacle_distance = 100.0  # Far away
        else:
            obstacle_location, obstacle_distance, _ = nearest_obstacle
        
        # Normalize obstacle distance (assuming max detection range of 50 meters)
        normalized_distance = min(obstacle_distance / 50.0, 1.0)
        
        # Calculate relative position of obstacle
        relative_position = obstacle_location - vehicle_location
        relative_position = relative_position / np.linalg.norm(relative_position) if np.linalg.norm(relative_position) > 0 else np.zeros(3)
        
        # Create input features
        features = np.array([
            normalized_velocity,
            normalized_distance,
            relative_position[0],  # x
            relative_position[1],  # y
            relative_position[2]   # z
        ])
        
        return features
    
    def predict_control(self, 
                       vehicle_location: np.ndarray,
                       vehicle_velocity: float,
                       vehicle_rotation: np.ndarray,
                       obstacles: List[Tuple[np.ndarray, float, str]]) -> Tuple[float, float, float]:
        """
        Predict control actions based on current state and obstacles.
        
        Args:
            vehicle_location: Current vehicle location (x, y, z)
            vehicle_velocity: Current vehicle velocity
            vehicle_rotation: Current vehicle rotation (pitch, yaw, roll)
            obstacles: List of tuples containing (location, distance, type) for each obstacle
            
        Returns:
            Tuple[float, float, float]: Predicted throttle, brake, and steering values
        """
        # Preprocess input
        features = self.preprocess_input(vehicle_location, vehicle_velocity, vehicle_rotation, obstacles)
        
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
                    target_controls: Tuple[float, float, float]):
        """
        Update the models based on new experience.
        
        Args:
            vehicle_location: Current vehicle location (x, y, z)
            vehicle_velocity: Current vehicle velocity
            vehicle_rotation: Current vehicle rotation (pitch, yaw, roll)
            obstacles: List of tuples containing (location, distance, type) for each obstacle
            target_controls: Target control values (throttle, brake, steer)
        """
        # Preprocess input
        features = self.preprocess_input(vehicle_location, vehicle_velocity, vehicle_rotation, obstacles)
        
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