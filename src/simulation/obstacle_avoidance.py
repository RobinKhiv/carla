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
        """Initialize models with basic rules for speed control."""
        # Create some basic training data
        X = []
        y_throttle = []
        y_brake = []
        y_steer = []
        
        # Case 1: Low speed
        X.append([0.2, 1.0, 0.0, 0.0, 0.0])  # low speed, no obstacles
        y_throttle.append(0.5)  # accelerate
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 2: Medium speed
        X.append([0.5, 1.0, 0.0, 0.0, 0.0])  # medium speed, no obstacles
        y_throttle.append(0.3)  # maintain speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 3: High speed
        X.append([0.8, 1.0, 0.0, 0.0, 0.0])  # high speed, no obstacles
        y_throttle.append(0.1)  # reduce speed
        y_brake.append(0.0)     # no brake
        y_steer.append(0.0)     # go straight
        
        # Case 4: Very high speed
        X.append([1.0, 1.0, 0.0, 0.0, 0.0])  # very high speed, no obstacles
        y_throttle.append(0.0)  # no throttle
        y_brake.append(0.2)     # light brake
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
        
        # For now, ignore obstacles and just focus on speed control
        # Create input features with just velocity
        features = np.array([
            normalized_velocity,
            1.0,  # default distance (no obstacles)
            0.0,  # default x position
            0.0,  # default y position
            0.0   # default z position
        ])
        
        return features
    
    def predict_control(self, 
                       vehicle_location: np.ndarray,
                       vehicle_velocity: float,
                       vehicle_rotation: np.ndarray,
                       obstacles: List[Tuple[np.ndarray, float, str]]) -> Tuple[float, float, float]:
        """
        Predict control actions based on current state.
        
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
        steer = 0.0  # Always go straight for now
        
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