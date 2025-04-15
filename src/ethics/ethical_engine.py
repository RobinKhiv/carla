from typing import Dict, Any, List, Tuple
from ..utils.sensor_utils import SensorUtils
import numpy as np

class EthicalEngine:
    def __init__(self):
        """Initialize the ethical engine with decision-making parameters."""
        self.sensor_utils = SensorUtils()
        self.ethical_priorities = {
            'pedestrian_safety': 1.0,
            'passenger_safety': 0.9,
            'other_vehicle_safety': 0.8,
            'property_damage': 0.5,
            'traffic_rules': 0.7
        }
        self.trolley_scenarios = []
        self.current_hazard = None

    def evaluate_decision(self, decision: Dict[str, Any], 
                         sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate and modify decisions based on ethical considerations."""
        # Get the base decision
        ethical_decision = decision.copy()
        
        # Get controls from the decision dictionary
        controls = ethical_decision.get('controls', {})
        throttle = controls.get('throttle', 0.0)
        brake = controls.get('brake', 0.0)
        steer = controls.get('steer', 0.0)
        
        # Get ethical weights
        ethical_weights = ethical_decision.get('ethical_weights', {})
        pedestrian_safety = ethical_weights.get('pedestrian_safety', 0.0)
        passenger_safety = ethical_weights.get('passenger_safety', 0.0)
        other_vehicle_safety = ethical_weights.get('other_vehicle_safety', 0.0)
        property_damage = ethical_weights.get('property_damage', 0.0)
        traffic_rules = ethical_weights.get('traffic_rules', 0.0)
        
        # Get trolley decision
        trolley_decision = ethical_decision.get('trolley_decision', {})
        continue_straight = trolley_decision.get('continue_straight', 0.0)
        swerve_left = trolley_decision.get('swerve_left', 0.0)
        swerve_right = trolley_decision.get('swerve_right', 0.0)
        
        # Apply ethical modifications
        if pedestrian_safety > 0.7:  # High pedestrian safety priority
            throttle *= 0.5
            brake *= 1.2
        
        if passenger_safety > 0.7:  # High passenger safety priority
            throttle *= 0.8
            brake *= 1.1
        
        if other_vehicle_safety > 0.7:  # High other vehicle safety priority
            throttle *= 0.7
            brake *= 1.1
        
        if property_damage > 0.7:  # High property damage priority
            throttle *= 0.6
            brake *= 1.2
        
        if traffic_rules > 0.7:  # High traffic rules priority
            throttle *= 0.9
            brake *= 1.1
        
        # Update the controls in the decision
        ethical_decision['controls'] = {
    def evaluate_decision(self, decision: Dict[str, Any], sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a decision based on ethical considerations."""
        # Calculate risk scores
        risk_score = self.sensor_utils.calculate_risk_score(sensor_data)
        
        # Detect obstacles
        obstacles = []
        if 'lidar' in sensor_data:
            obstacles = self.sensor_utils.detect_obstacles(sensor_data['lidar'])
        
        # Apply ethical weights to the decision
        ethical_decision = decision.copy()
        
        # Adjust controls based on risk
        if risk_score > 0.5:
            # High risk situation
            ethical_decision['throttle'] *= 0.5
            ethical_decision['brake'] = max(decision['brake'], 0.5)
        elif risk_score > 0.3:
            # Moderate risk
            ethical_decision['throttle'] *= 0.7
            ethical_decision['brake'] = max(decision['brake'], 0.3)
        
        # Check for pedestrians
        if obstacles:
            for obstacle in obstacles:
                if obstacle['size'] > 1.0:  # Significant obstacle
                    # Strong braking for large obstacles
                    ethical_decision['throttle'] = 0.0
                    ethical_decision['brake'] = 1.0
                    break
        
        return ethical_decision

    def _check_trolley_problem(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for trolley problem scenarios and make ethical decisions.
        
        A trolley problem occurs when the vehicle must choose between:
        1. Continuing on current path (potentially harming multiple people)
        2. Taking evasive action (potentially harming fewer people or property)
        """
        # Get detected objects
        objects = sensor_data.get('objects', [])
        
        # Check for multiple pedestrians in danger
        pedestrians_in_danger = [obj for obj in objects 
                               if obj['type'] == 'pedestrian' 
                               and obj['distance'] < 20.0]
        
        if len(pedestrians_in_danger) >= 2:
            # Calculate potential outcomes
            current_path_risk = self._calculate_risk(pedestrians_in_danger)
            alternative_path = self._find_alternative_path(sensor_data)
            alternative_risk = self._calculate_risk(alternative_path['objects'])
            
            # Make decision based on ethical framework
            if alternative_risk < current_path_risk:
                return {
                    'action': 'swerve',
                    'direction': alternative_path['direction'],
                    'throttle': 0.5,
                    'brake': 0.0,
                    'steer': alternative_path['steer'],
                    'ethical_justification': 'Minimizing harm by choosing path with fewer potential casualties'
                }
        
        return None

    def _check_hazards(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for hazardous situations and respond appropriately.
        """
        objects = sensor_data.get('objects', [])
        
        # Check for immediate hazards
        immediate_hazards = [obj for obj in objects 
                           if obj['distance'] < 10.0 
                           and (obj['type'] == 'pedestrian' or obj['type'] == 'vehicle')]
        
        if immediate_hazards:
            # Calculate risk level
            risk_level = self._calculate_risk(immediate_hazards)
            
            if risk_level > 0.7:  # High risk situation
                return {
                    'action': 'emergency_brake',
                    'throttle': 0.0,
                    'brake': 1.0,
                    'steer': 0.0,
                    'ethical_justification': 'Emergency stop to prevent collision with immediate hazard'
                }
            elif risk_level > 0.3:  # Medium risk situation
                return {
                    'action': 'slow_down',
                    'throttle': 0.2,
                    'brake': 0.5,
                    'steer': 0.0,
                    'ethical_justification': 'Reducing speed to safely navigate hazard'
                }
        
        return None

    def _calculate_risk(self, objects: List[Dict[str, Any]]) -> float:
        """
        Calculate the risk level based on detected objects.
        """
        if not objects:
            return 0.0
        
        total_risk = 0.0
        for obj in objects:
            # Risk increases with proximity and object type
            proximity_risk = 1.0 - (obj['distance'] / 50.0)  # Normalize distance to 0-1
            type_risk = self.ethical_priorities.get(f"{obj['type']}_safety", 0.5)
            total_risk += proximity_risk * type_risk
        
        return min(1.0, total_risk / len(objects))

    def _find_alternative_path(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find an alternative path with lower risk.
        """
        # Get current lane information
        current_lane = sensor_data.get('lane_info', {})
        
        # Check left and right lanes
        left_lane_clear = self._is_lane_clear(sensor_data, 'left')
        right_lane_clear = self._is_lane_clear(sensor_data, 'right')
        
        if left_lane_clear and not right_lane_clear:
            return {
                'direction': 'left',
                'steer': -0.5,
                'objects': []
            }
        elif right_lane_clear and not left_lane_clear:
            return {
                'direction': 'right',
                'steer': 0.5,
                'objects': []
            }
        elif left_lane_clear and right_lane_clear:
            # Choose the lane with more space
            left_space = self._calculate_lane_space(sensor_data, 'left')
            right_space = self._calculate_lane_space(sensor_data, 'right')
            if left_space > right_space:
                return {
                    'direction': 'left',
                    'steer': -0.5,
                    'objects': []
                }
            else:
                return {
                    'direction': 'right',
                    'steer': 0.5,
                    'objects': []
                }
        
        # If no clear alternative, return current path
        return {
            'direction': 'straight',
            'steer': 0.0,
            'objects': sensor_data.get('objects', [])
        }

    def _is_lane_clear(self, sensor_data: Dict[str, Any], direction: str) -> bool:
        """
        Check if a lane is clear of obstacles.
        """
        objects = sensor_data.get('objects', [])
        lane_objects = [obj for obj in objects 
                       if obj['lane'] == direction 
                       and obj['distance'] < 20.0]
        return len(lane_objects) == 0

    def _calculate_lane_space(self, sensor_data: Dict[str, Any], direction: str) -> float:
        """
        Calculate the available space in a lane.
        """
        objects = sensor_data.get('objects', [])
        lane_objects = [obj for obj in objects if obj['lane'] == direction]
        if not lane_objects:
            return float('inf')
        return min(obj['distance'] for obj in lane_objects)

    def _apply_ethical_constraints(self, decision: Dict[str, Any], sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply standard ethical constraints to the decision.
        """
        # Get current speed and distance to nearest object
        current_speed = sensor_data.get('speed', 0.0)
        nearest_object = min(sensor_data.get('objects', []), 
                           key=lambda x: x['distance'], 
                           default={'distance': float('inf')})
        
        # Adjust speed based on proximity to objects
        if nearest_object['distance'] < 20.0:
            decision['throttle'] = min(decision['throttle'], 0.5)
            decision['brake'] = max(decision['brake'], 0.2)
        
        # Ensure safe following distance
        if nearest_object['distance'] < 10.0:
            decision['throttle'] = 0.0
            decision['brake'] = 1.0
        
        return decision

    def _detect_pedestrians(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect pedestrians from sensor data."""
        pedestrians = []
        
        if 'lidar' in sensor_data:
            obstacles = self.sensor_utils.detect_obstacles(sensor_data['lidar'])
            for obstacle in obstacles:
                # Simple heuristic: pedestrians are typically smaller than vehicles
                if obstacle['size'] < 1.5:  # Approximate pedestrian size
                    pedestrians.append({
                        'position': obstacle['center'],
                        'distance': np.linalg.norm(obstacle['center']),
                        'direction': np.arctan2(obstacle['center'][1], obstacle['center'][0])
                    })
        
        return pedestrians

    def _has_pedestrians_in_path(self, sensor_data: Dict[str, Any]) -> bool:
        """Check if there are pedestrians in the vehicle's path."""
        pedestrians = self._detect_pedestrians(sensor_data)
        if not pedestrians:
            return False
        
        # Check if any pedestrian is in front of the vehicle
        for pedestrian in pedestrians:
            if abs(pedestrian['direction']) < np.pi/4:  # Within 45 degrees of forward
                return True
        
        return False

    def update_priorities(self, pedestrian_priority: float = None,
                         passenger_priority: float = None,
                         property_priority: float = None):
        """Update ethical priorities."""
        if pedestrian_priority is not None:
            self.pedestrian_priority = pedestrian_priority
        if passenger_priority is not None:
            self.passenger_priority = passenger_priority
        if property_priority is not None:
            self.property_priority = property_priority

    def get_ethical_score(self, decision: Dict[str, float], 
                         sensor_data: Dict[str, Any]) -> float:
        """Calculate an ethical score for a decision."""
        score = 0.0
        
        # Check for pedestrian safety
        pedestrians = self._detect_pedestrians(sensor_data)
        if pedestrians:
            closest_pedestrian = min(pedestrians, key=lambda x: x['distance'])
            if closest_pedestrian['distance'] < self.min_safe_distance:
                score -= self.pedestrian_priority
        
        # Check for passenger safety
        risk_score = self.sensor_utils.calculate_risk_score(sensor_data)
        if risk_score > self.emergency_braking_threshold:
            score -= self.passenger_priority
        
        # Check for property damage
        if 'collision' in sensor_data:
            score -= self.property_priority
        
        return score 