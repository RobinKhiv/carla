import random
from typing import Dict, Any, List, Tuple
import numpy as np
from carla import Transform, Location, Rotation
from .rl_components import RLManager

class PedestrianTrainingScenarios:
    """Class for generating and managing pedestrian safety training scenarios."""
    
    def __init__(self, world, rl_manager: RLManager):
        self.world = world
        self.rl_manager = rl_manager
        self.scenarios = []
        self.current_scenario = None
        
        # Initialize scenarios
        self._initialize_scenarios()
        
    def _initialize_scenarios(self):
        """Initialize the set of training scenarios."""
        self.scenarios = [
            {
                'name': 'pedestrian_crossing',
                'description': 'Pedestrian crossing in front of vehicle',
                'setup': self._setup_pedestrian_crossing,
                'success_criteria': self._check_pedestrian_crossing_success
            },
            {
                'name': 'pedestrian_walking_parallel',
                'description': 'Pedestrian walking parallel to vehicle path',
                'setup': self._setup_parallel_pedestrian,
                'success_criteria': self._check_parallel_pedestrian_success
            },
            {
                'name': 'multiple_pedestrians',
                'description': 'Multiple pedestrians in different positions',
                'setup': self._setup_multiple_pedestrians,
                'success_criteria': self._check_multiple_pedestrians_success
            },
            {
                'name': 'pedestrian_sudden_crossing',
                'description': 'Pedestrian suddenly crossing from sidewalk',
                'setup': self._setup_sudden_crossing,
                'success_criteria': self._check_sudden_crossing_success
            },
            {
                'name': 'pedestrian_group_crossing',
                'description': 'Group of pedestrians crossing together',
                'setup': self._setup_group_crossing,
                'success_criteria': self._check_group_crossing_success
            },
            {
                'name': 'emergency_swerve',
                'description': 'Pedestrian suddenly appears in front of vehicle',
                'setup': self._setup_emergency_swerve,
                'success_criteria': self._check_emergency_swerve_success
            },
            {
                'name': 'obstacle_avoidance',
                'description': 'Multiple obstacles requiring swerving',
                'setup': self._setup_obstacle_avoidance,
                'success_criteria': self._check_obstacle_avoidance_success
            },
            {
                'name': 'split_second_decision',
                'description': 'Multiple pedestrians appearing suddenly',
                'setup': self._setup_split_second_decision,
                'success_criteria': self._check_split_second_decision_success
            }
        ]
        
    def select_scenario(self) -> Dict[str, Any]:
        """Select a random training scenario."""
        self.current_scenario = random.choice(self.scenarios)
        return self.current_scenario
        
    def setup_scenario(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Set up the current scenario and return initial state."""
        if not self.current_scenario:
            self.select_scenario()
            
        # Get spawn points for vehicle and pedestrians
        vehicle_spawn, pedestrian_spawns = self.current_scenario['setup']()
        
        # Create initial state
        initial_state = {
            'scenario_name': self.current_scenario['name'],
            'pedestrian_count': len(pedestrian_spawns),
            'vehicle_position': vehicle_spawn.location,
            'pedestrian_positions': [p.location for p in pedestrian_spawns],
            'time_elapsed': 0.0,
            'success': False,
            'failure_reason': None
        }
        
        return pedestrian_spawns, initial_state
        
    def _setup_pedestrian_crossing(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario with pedestrian crossing in front of vehicle."""
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        # Calculate pedestrian spawn point in front of vehicle
        pedestrian_location = Location(
            x=vehicle_spawn.location.x + 10.0,
            y=vehicle_spawn.location.y,
            z=vehicle_spawn.location.z
        )
        pedestrian_spawn = Transform(
            location=pedestrian_location,
            rotation=Rotation(pitch=0, yaw=90, roll=0)
        )
        
        return vehicle_spawn, [pedestrian_spawn]
        
    def _setup_parallel_pedestrian(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario with pedestrian walking parallel to vehicle path."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        # Calculate pedestrian spawn point parallel to vehicle path
        pedestrian_location = Location(
            x=vehicle_spawn.location.x,
            y=vehicle_spawn.location.y + 3.0,  # 3 meters to the side
            z=vehicle_spawn.location.z
        )
        pedestrian_spawn = Transform(
            location=pedestrian_location,
            rotation=Rotation(pitch=0, yaw=vehicle_spawn.rotation.yaw, roll=0)
        )
        
        return vehicle_spawn, [pedestrian_spawn]
        
    def _setup_multiple_pedestrians(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario with multiple pedestrians in different positions."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        pedestrian_spawns = []
        for i in range(3):  # Create 3 pedestrians
            angle = random.uniform(0, 360)
            distance = random.uniform(5.0, 15.0)
            
            pedestrian_location = Location(
                x=vehicle_spawn.location.x + distance * np.cos(np.radians(angle)),
                y=vehicle_spawn.location.y + distance * np.sin(np.radians(angle)),
                z=vehicle_spawn.location.z
            )
            pedestrian_spawn = Transform(
                location=pedestrian_location,
                rotation=Rotation(pitch=0, yaw=angle, roll=0)
            )
            pedestrian_spawns.append(pedestrian_spawn)
            
        return vehicle_spawn, pedestrian_spawns
        
    def _setup_sudden_crossing(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario with pedestrian suddenly crossing from sidewalk."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        # Calculate pedestrian spawn point on sidewalk
        pedestrian_location = Location(
            x=vehicle_spawn.location.x + 15.0,
            y=vehicle_spawn.location.y + 5.0,  # On sidewalk
            z=vehicle_spawn.location.z
        )
        pedestrian_spawn = Transform(
            location=pedestrian_location,
            rotation=Rotation(pitch=0, yaw=270, roll=0)  # Facing road
        )
        
        return vehicle_spawn, [pedestrian_spawn]
        
    def _setup_group_crossing(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario with group of pedestrians crossing together."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        pedestrian_spawns = []
        for i in range(5):  # Create group of 5 pedestrians
            offset = random.uniform(-2.0, 2.0)
            pedestrian_location = Location(
                x=vehicle_spawn.location.x + 10.0,
                y=vehicle_spawn.location.y + offset,
                z=vehicle_spawn.location.z
            )
            pedestrian_spawn = Transform(
                location=pedestrian_location,
                rotation=Rotation(pitch=0, yaw=90, roll=0)
            )
            pedestrian_spawns.append(pedestrian_spawn)
            
        return vehicle_spawn, pedestrian_spawns
        
    def _setup_emergency_swerve(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario requiring emergency swerve maneuver."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        # Calculate pedestrian spawn point very close to vehicle path
        pedestrian_location = Location(
            x=vehicle_spawn.location.x + 5.0,  # Very close to vehicle
            y=vehicle_spawn.location.y,
            z=vehicle_spawn.location.z
        )
        pedestrian_spawn = Transform(
            location=pedestrian_location,
            rotation=Rotation(pitch=0, yaw=90, roll=0)
        )
        
        return vehicle_spawn, [pedestrian_spawn]
        
    def _setup_obstacle_avoidance(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario with multiple obstacles requiring swerving."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        pedestrian_spawns = []
        # Create obstacles in a zigzag pattern
        for i in range(3):
            offset = 2.0 if i % 2 == 0 else -2.0
            pedestrian_location = Location(
                x=vehicle_spawn.location.x + (i + 1) * 5.0,
                y=vehicle_spawn.location.y + offset,
                z=vehicle_spawn.location.z
            )
            pedestrian_spawn = Transform(
                location=pedestrian_location,
                rotation=Rotation(pitch=0, yaw=90, roll=0)
            )
            pedestrian_spawns.append(pedestrian_spawn)
            
        return vehicle_spawn, pedestrian_spawns
        
    def _setup_split_second_decision(self) -> Tuple[Transform, List[Transform]]:
        """Set up scenario requiring quick decision between multiple options."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_spawn = random.choice(spawn_points)
        
        pedestrian_spawns = []
        # Create pedestrians in a V-pattern
        for i in range(3):
            angle = 45 if i == 0 else -45 if i == 2 else 0
            distance = 7.0
            pedestrian_location = Location(
                x=vehicle_spawn.location.x + distance,
                y=vehicle_spawn.location.y + distance * np.tan(np.radians(angle)),
                z=vehicle_spawn.location.z
            )
            pedestrian_spawn = Transform(
                location=pedestrian_location,
                rotation=Rotation(pitch=0, yaw=angle, roll=0)
            )
            pedestrian_spawns.append(pedestrian_spawn)
            
        return vehicle_spawn, pedestrian_spawns
        
    def update_scenario(self, state: Dict[str, Any], 
                       vehicle_state: Dict[str, Any],
                       pedestrian_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update scenario state and check success/failure conditions."""
        state['time_elapsed'] += 0.1  # Assuming 0.1s time step
        
        # Check success criteria based on current scenario
        success, failure_reason = self.current_scenario['success_criteria'](
            state, vehicle_state, pedestrian_states
        )
        
        state['success'] = success
        state['failure_reason'] = failure_reason
        
        return state
        
    def _check_pedestrian_crossing_success(self, state: Dict[str, Any],
                                         vehicle_state: Dict[str, Any],
                                         pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for pedestrian crossing scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        if vehicle_state['speed'] > 2.0 and pedestrian_states[0]['distance'] < 5.0:
            return False, "Vehicle too fast near pedestrian"
            
        if state['time_elapsed'] > 30.0:
            return True, "Scenario completed successfully"
            
        return False, None
        
    def _check_parallel_pedestrian_success(self, state: Dict[str, Any],
                                         vehicle_state: Dict[str, Any],
                                         pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for parallel pedestrian scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        if vehicle_state['speed'] > 3.0 and pedestrian_states[0]['distance'] < 3.0:
            return False, "Vehicle too fast near parallel pedestrian"
            
        if state['time_elapsed'] > 20.0:
            return True, "Scenario completed successfully"
            
        return False, None
        
    def _check_multiple_pedestrians_success(self, state: Dict[str, Any],
                                          vehicle_state: Dict[str, Any],
                                          pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for multiple pedestrians scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        for ped_state in pedestrian_states:
            if vehicle_state['speed'] > 2.0 and ped_state['distance'] < 5.0:
                return False, "Vehicle too fast near pedestrian"
                
        if state['time_elapsed'] > 40.0:
            return True, "Scenario completed successfully"
            
        return False, None
        
    def _check_sudden_crossing_success(self, state: Dict[str, Any],
                                     vehicle_state: Dict[str, Any],
                                     pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for sudden crossing scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        if vehicle_state['speed'] > 1.5 and pedestrian_states[0]['distance'] < 7.0:
            return False, "Vehicle too fast near suddenly crossing pedestrian"
            
        if state['time_elapsed'] > 25.0:
            return True, "Scenario completed successfully"
            
        return False, None
        
    def _check_group_crossing_success(self, state: Dict[str, Any],
                                    vehicle_state: Dict[str, Any],
                                    pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for group crossing scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        for ped_state in pedestrian_states:
            if vehicle_state['speed'] > 1.0 and ped_state['distance'] < 8.0:
                return False, "Vehicle too fast near pedestrian group"
                
        if state['time_elapsed'] > 35.0:
            return True, "Scenario completed successfully"
            
        return False, None
        
    def _check_emergency_swerve_success(self, state: Dict[str, Any],
                                      vehicle_state: Dict[str, Any],
                                      pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for emergency swerve scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        # Check if vehicle performed emergency maneuver
        if vehicle_state['steering'] < 0.3 and vehicle_state['speed'] > 2.0:
            return False, "No emergency maneuver performed"
            
        # Check if vehicle maintained control
        if abs(vehicle_state['steering']) > 0.8:
            return False, "Vehicle lost control during maneuver"
            
        if state['time_elapsed'] > 15.0:
            return True, "Emergency maneuver completed successfully"
            
        return False, None
        
    def _check_obstacle_avoidance_success(self, state: Dict[str, Any],
                                        vehicle_state: Dict[str, Any],
                                        pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for obstacle avoidance scenario."""
        if vehicle_state['collision']:
            return False, "Collision with obstacle"
            
        # Check if vehicle performed necessary maneuvers
        if abs(vehicle_state['steering']) < 0.2 and vehicle_state['speed'] > 3.0:
            return False, "No obstacle avoidance maneuvers performed"
            
        # Check if vehicle maintained safe distance
        for ped_state in pedestrian_states:
            if ped_state['distance'] < 2.0:
                return False, "Vehicle too close to obstacle"
                
        if state['time_elapsed'] > 25.0:
            return True, "Obstacle avoidance completed successfully"
            
        return False, None
        
    def _check_split_second_decision_success(self, state: Dict[str, Any],
                                           vehicle_state: Dict[str, Any],
                                           pedestrian_states: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check success criteria for split-second decision scenario."""
        if vehicle_state['collision']:
            return False, "Collision with pedestrian"
            
        # Check if vehicle made a clear decision
        if abs(vehicle_state['steering']) < 0.1 and vehicle_state['speed'] > 2.0:
            return False, "No clear decision made"
            
        # Check if vehicle chose the safest path
        min_distance = min(ped_state['distance'] for ped_state in pedestrian_states)
        if min_distance < 3.0:
            return False, "Vehicle chose unsafe path"
            
        if state['time_elapsed'] > 20.0:
            return True, "Split-second decision completed successfully"
            
        return False, None 