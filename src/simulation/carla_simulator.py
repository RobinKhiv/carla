import carla
import time
import math
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
from .obstacle_avoidance import ObstacleAvoidance
from .rl_agent import EthicalPriorities
from .rl_agent import RLEthicalAgent

class CarlaSimulator:
    def __init__(self, host: str = 'localhost', port: int = 2000):
        """Initialize the CARLA simulator with connection parameters."""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = None
            self.vehicle = None
            self.spectator = None
            self.running = False
            self.traffic_manager = None
            self.initialized = False
            self.obstacle_avoidance = ObstacleAvoidance()
            
            # Initialize reinforcement learning agent with ethical priorities
            ethical_priorities = EthicalPriorities(
                pedestrian_weight=1.0,  # High priority for pedestrian safety
                passenger_weight=1.0,   # Equal priority for passenger safety
                property_weight=0.5,    # Lower priority for property damage
                traffic_law_weight=0.8  # High priority for traffic laws
            )
            self.rl_agent = RLEthicalAgent(
                state_size=5,  # Speed, nearby actors, traffic light, next waypoint x, y
                action_size=9,  # 3 throttle levels × 3 steering levels
                ethical_priorities=ethical_priorities
            )
            
        except Exception as e:
            print(f"Error initializing CARLA simulator: {e}")
            raise

    def initialize(self):
        """Initialize the simulation environment and components."""
        try:
            # Check if CARLA server is running
            print("Connecting to CARLA server...")
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            
            # Get available maps
            available_maps = self.client.get_available_maps()
            print(f"Available maps: {available_maps}")
            
            if not available_maps:
                raise RuntimeError("No maps available in CARLA server")
            
            # Try to load Town03, fall back to Town01 if not available
            try:
                print("Loading Town03 map...")
                self.client.load_world('Town03')
            except Exception as e:
                print(f"Failed to load Town03, trying Town01: {e}")
                self.client.load_world('Town01')
            
            # Get the world
            self.world = self.client.get_world()
            if self.world is None:
                raise RuntimeError("Failed to get CARLA world")
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            # Initialize traffic manager with hybrid physics mode
            print("Initializing traffic manager...")
            self.traffic_manager = self.client.get_trafficmanager()
            if self.traffic_manager is None:
                raise RuntimeError("Failed to get traffic manager")
            
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
            
            print("Simulator initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            print("Please ensure that:")
            print("1. CARLA server is running")
            print("2. CARLA server is accessible at localhost:2000")
            print("3. You have the correct version of CARLA installed")
            return False

    def spawn_vehicle(self) -> bool:
        """Spawn the ego vehicle."""
        try:
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points available")
                return False
            
            # Try different spawn points until successful
            for spawn_point in spawn_points:
                try:
                    # Check if spawn point is on the road
                    waypoint = self.world.get_map().get_waypoint(spawn_point.location)
                    if waypoint is None:
                        continue  # Skip if not on road
                    
                    # Check for collisions at spawn point
                    collision = False
                    for actor in self.world.get_actors():
                        if actor.get_location().distance(spawn_point.location) < 10.0:
                            collision = True
                            break
                    
                    if not collision:
                        # Spawn vehicle
                        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                        if self.vehicle is not None:
                            print(f"Vehicle spawned successfully at {spawn_point.location}")
                            
                            # Set vehicle physics
                            self.vehicle.set_simulate_physics(True)
                            
                            # Set vehicle to autopilot with traffic manager
                            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
                            
                            # Configure traffic manager settings
                            self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)  # Always obey traffic lights
                            self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 0)  # Maintain normal speed
                            self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 2.0)  # Safe following distance
                            self.traffic_manager.auto_lane_change(self.vehicle, False)  # Disable automatic lane changes
                            self.traffic_manager.set_hybrid_physics_mode(True)  # Enable hybrid physics
                            self.traffic_manager.set_hybrid_physics_radius(70.0)  # Set physics radius
                            
                            print("Vehicle registered with traffic manager")
                            return True
                except Exception as e:
                    print(f"Failed to spawn at point {spawn_point.location}: {e}")
                    continue
            
            print("Failed to find valid spawn point")
            return False
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return False

    def setup_camera(self):
        """Set up the spectator camera to follow the vehicle."""
        try:
            if not self.vehicle:
                print("Warning: Cannot setup camera - no vehicle available")
                return False

            # Get the spectator
            self.spectator = self.world.get_spectator()
            if not self.spectator:
                print("Warning: Failed to get spectator")
                return False

            # Get vehicle transform
            vehicle_transform = self.vehicle.get_transform()
            
            # Calculate camera position (behind and above the vehicle)
            camera_location = carla.Location(
                x=vehicle_transform.location.x - 10.0 * math.cos(math.radians(vehicle_transform.rotation.yaw)),
                y=vehicle_transform.location.y - 10.0 * math.sin(math.radians(vehicle_transform.rotation.yaw)),
                z=vehicle_transform.location.z + 5.0
            )
            
            # Set camera rotation to look at vehicle
            camera_rotation = carla.Rotation(
                pitch=-15.0,
                yaw=vehicle_transform.rotation.yaw,
                roll=0.0
            )
            
            # Set spectator transform
            self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))
            print("Camera setup complete")
            return True
        except Exception as e:
            print(f"Error setting up camera: {e}")
            return False

    def update_camera(self):
        """Update camera position and orientation"""
        if self.vehicle and self.spectator:
            try:
                # Get vehicle transform
                vehicle_transform = self.vehicle.get_transform()
                
                # Calculate camera position (behind and above the vehicle)
                camera_location = carla.Location(
                    x=vehicle_transform.location.x - 10.0 * math.cos(math.radians(vehicle_transform.rotation.yaw)),
                    y=vehicle_transform.location.y - 10.0 * math.sin(math.radians(vehicle_transform.rotation.yaw)),
                    z=vehicle_transform.location.z + 5.0
                )
                
                # Calculate camera rotation (looking at vehicle)
                camera_rotation = carla.Rotation(
                    pitch=-15.0,
                    yaw=vehicle_transform.rotation.yaw,
                    roll=0.0
                )
                
                # Set camera transform
                self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))
                
            except Exception as e:
                print(f"Error updating camera: {e}")

    def spawn_traffic(self, num_vehicles: int = 10, num_pedestrians: int = 50):
        """Spawn traffic and create specific scenarios for testing ethical decision making."""
        try:
            # Get vehicle blueprints
            vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points available for vehicles")
                return
            
            # Configure traffic manager
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(0)
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
            
            # Spawn vehicles
            for i in range(min(num_vehicles, len(spawn_points))):
                try:
                    # Check for collisions at spawn point
                    collision = False
                    for actor in self.world.get_actors():
                        if actor.get_location().distance(spawn_points[i].location) < 5.0:
                            collision = True
                            break
                    
                    if not collision:
                        vehicle = self.world.spawn_actor(
                            random.choice(vehicle_bp),
                            spawn_points[i]
                        )
                        if vehicle is not None:
                            # Set autopilot with traffic manager
                            vehicle.set_autopilot(True, self.traffic_manager.get_port())
                            # Set speed limit
                            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 30.0)  # 30% slower
                            # Set collision detection
                            self.traffic_manager.auto_lane_change(vehicle, False)
                            self.traffic_manager.distance_to_leading_vehicle(vehicle, 2.0)
                except Exception as e:
                    print(f"Warning: Failed to spawn vehicle at point {i}: {e}")
                    continue
            
            # Create specific pedestrian scenarios
            try:
                # Get pedestrian blueprints
                walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
                
                # Get all waypoints in the map
                waypoints = self.world.get_map().generate_waypoints(1.0)  # 1.0 meters apart
                
                # Filter waypoints that are on sidewalks
                sidewalk_waypoints = [wp for wp in waypoints if wp.is_junction or wp.lane_type == carla.LaneType.Sidewalk]
                
                if not sidewalk_waypoints:
                    print("Warning: No sidewalk waypoints found")
                    return
                
                # Scenario 1: Crosswalk scenario
                crosswalk_waypoints = [wp for wp in sidewalk_waypoints if wp.is_junction]
                if crosswalk_waypoints:
                    for i in range(10):  # 10 pedestrians at crosswalk
                        max_retries = 5
                        for retry in range(max_retries):
                            try:
                                # Select a random crosswalk waypoint
                                spawn_waypoint = random.choice(crosswalk_waypoints)
                                spawn_point = carla.Transform()
                                spawn_point.location = spawn_waypoint.transform.location
                                spawn_point.location.z += 0.5  # Raise slightly above ground
                                
                                # Check for collisions before spawning
                                collision = False
                                for actor in self.world.get_actors():
                                    if actor.get_location().distance(spawn_point.location) < 2.0:
                                        collision = True
                                        break
                                
                                if not collision:
                                    walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                                    if walker is not None:
                                        # Find a destination waypoint across the street
                                        next_waypoints = spawn_waypoint.next(5.0)
                                        if next_waypoints:
                                            destination = random.choice(next_waypoints).transform.location
                                            destination.z = spawn_point.location.z
                                            walker.set_location(destination)
                                            self.pedestrians.append(walker)
                                            break
                            except Exception as e:
                                if retry == max_retries - 1:
                                    print(f"Warning: Failed to spawn crosswalk pedestrian after {max_retries} attempts: {e}")
                
                # Scenario 2: School zone scenario
                school_zone_waypoints = [wp for wp in sidewalk_waypoints if not wp.is_junction]
                if school_zone_waypoints:
                    for i in range(15):  # 15 pedestrians in school zone
                        max_retries = 5
                        for retry in range(max_retries):
                            try:
                                # Select a random sidewalk waypoint
                                spawn_waypoint = random.choice(school_zone_waypoints)
                                spawn_point = carla.Transform()
                                spawn_point.location = spawn_waypoint.transform.location
                                spawn_point.location.z += 0.5  # Raise slightly above ground
                                
                                # Check for collisions before spawning
                                collision = False
                                for actor in self.world.get_actors():
                                    if actor.get_location().distance(spawn_point.location) < 2.0:
                                        collision = True
                                        break
                                
                                if not collision:
                                    walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                                    if walker is not None:
                                        # Find a destination waypoint along the sidewalk
                                        next_waypoints = spawn_waypoint.next(10.0)
                                        if next_waypoints:
                                            destination = random.choice(next_waypoints).transform.location
                                            destination.z = spawn_point.location.z
                                            walker.set_location(destination)
                                            self.pedestrians.append(walker)
                                            break
                            except Exception as e:
                                if retry == max_retries - 1:
                                    print(f"Warning: Failed to spawn school zone pedestrian after {max_retries} attempts: {e}")
                
                # Scenario 3: Busy intersection scenario
                intersection_waypoints = [wp for wp in sidewalk_waypoints if wp.is_junction]
                if intersection_waypoints:
                    for i in range(25):  # 25 pedestrians at intersection
                        max_retries = 5
                        for retry in range(max_retries):
                            try:
                                # Select a random intersection waypoint
                                spawn_waypoint = random.choice(intersection_waypoints)
                                spawn_point = carla.Transform()
                                spawn_point.location = spawn_waypoint.transform.location
                                spawn_point.location.z += 0.5  # Raise slightly above ground
                                
                                # Check for collisions before spawning
                                collision = False
                                for actor in self.world.get_actors():
                                    if actor.get_location().distance(spawn_point.location) < 2.0:
                                        collision = True
                                        break
                                
                                if not collision:
                                    walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                                    if walker is not None:
                                        # Find a destination waypoint in a random direction
                                        next_waypoints = spawn_waypoint.next(15.0)
                                        if next_waypoints:
                                            destination = random.choice(next_waypoints).transform.location
                                            destination.z = spawn_point.location.z
                                            walker.set_location(destination)
                                            self.pedestrians.append(walker)
                                            break
                            except Exception as e:
                                if retry == max_retries - 1:
                                    print(f"Warning: Failed to spawn intersection pedestrian after {max_retries} attempts: {e}")
                
                print(f"Spawned {num_vehicles} vehicles and {len(self.pedestrians)} pedestrians in various scenarios")
            except Exception as e:
                print(f"Error spawning pedestrians: {e}")
            
        except Exception as e:
            print(f"Error spawning traffic: {e}")

    def detect_obstacles(self) -> List[Tuple[np.ndarray, float, str]]:
        """Detect obstacles (vehicles and pedestrians) around the ego vehicle."""
        obstacles = []
        
        if not self.vehicle:
            return obstacles
        
        # Get vehicle location and transform
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        # Get all actors in the world
        actors = self.world.get_actors()
        
        # Check for vehicles
        for actor in actors.filter('vehicle.*'):
            if actor.id != self.vehicle.id:  # Skip ego vehicle
                actor_location = actor.get_location()
                distance = vehicle_location.distance(actor_location)
                
                # Only consider vehicles in front of us
                vehicle_forward = vehicle_transform.get_forward_vector()
                actor_direction = actor_location - vehicle_location
                actor_direction = actor_direction.make_unit_vector()
                
                if vehicle_forward.dot(actor_direction) > 0.5:  # Only consider if in front
                    obstacles.append((
                        np.array([actor_location.x, actor_location.y, actor_location.z]),
                        distance,
                        'vehicle'
                    ))
        
        # Check for pedestrians
        for actor in actors.filter('walker.*'):
            actor_location = actor.get_location()
            distance = vehicle_location.distance(actor_location)
            
            # Only consider pedestrians in front of us
            vehicle_forward = vehicle_transform.get_forward_vector()
            actor_direction = actor_location - vehicle_location
            actor_direction = actor_direction.make_unit_vector()
            
            if vehicle_forward.dot(actor_direction) > 0.5:  # Only consider if in front
                obstacles.append((
                    np.array([actor_location.x, actor_location.y, actor_location.z]),
                    distance,
                    'pedestrian'
                ))
        
        return obstacles

    def run(self):
        """Run the simulation."""
        try:
            # Initialize simulator
            if not self.initialized:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize simulator")
                self.initialized = True
            
            # Clean up existing actors
            self.cleanup()
            
            # Spawn ego vehicle
            print("Spawning ego vehicle...")
            if not self.spawn_vehicle():
                raise RuntimeError("Failed to spawn ego vehicle")
            
            # Spawn traffic
            print("Spawning traffic...")
            self.spawn_traffic(10, 50)  # Spawn 10 vehicles and 50 pedestrians
            
            # Set up camera
            if not self.setup_camera():
                print("Warning: Camera setup failed, continuing without camera")
            
            # Configure traffic manager for ego vehicle
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)  # Always obey traffic lights
            self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 0)  # Maintain normal speed
            self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 2.0)  # Safe following distance
            self.traffic_manager.auto_lane_change(self.vehicle, True)  # Enable automatic lane changes
            self.traffic_manager.set_hybrid_physics_mode(True)  # Enable hybrid physics
            self.traffic_manager.set_hybrid_physics_radius(70.0)  # Set physics radius
            self.traffic_manager.random_left_lanechange_percentage(self.vehicle, 100)  # Allow left lane changes
            self.traffic_manager.random_right_lanechange_percentage(self.vehicle, 100)  # Allow right lane changes
            self.traffic_manager.keep_right_rule_percentage(self.vehicle, 0)  # Disable keep right rule
            
            # Main simulation loop
            self.running = True
            
            while self.running:
                try:
                    # Tick the world
                    self.world.tick()
                    
                    # Update camera position
                    self.update_camera()
                    
                    # Get vehicle state
                    try:
                        vehicle_location = self.vehicle.get_location()
                        vehicle_velocity = self.vehicle.get_velocity()
                        speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6  # Convert to km/h
                    except Exception as e:
                        print(f"Error getting vehicle state: {e}")
                        raise
                    
                    # Get current waypoint and next waypoints
                    try:
                        current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
                        if current_waypoint:
                            # Get next waypoints for navigation
                            next_waypoints = current_waypoint.next(5.0)  # Look 5 meters ahead
                            
                            if next_waypoints:
                                # Use CARLA's built-in navigation
                                next_waypoint = next_waypoints[0]
                                
                                # Check for obstacles using CARLA's built-in system
                                nearby_actors = self.world.get_actors()
                                
                                # Check for pedestrians in front
                                pedestrian_in_path = False
                                pedestrian_location = None
                                distance_to_pedestrian = float('inf')
                                
                                for actor in nearby_actors.filter('walker.*'):
                                    actor_location = actor.get_location()
                                    distance = float(vehicle_location.distance(actor_location))
                                    
                                    if distance < 20.0:  # Check within 20 meters
                                        # Calculate if pedestrian is in vehicle's path
                                        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
                                        actor_direction = actor_location - vehicle_location
                                        actor_direction = actor_direction.make_unit_vector()
                                        
                                        # If pedestrian is in front (within 30 degrees)
                                        if vehicle_forward.dot(actor_direction) > 0.866:  # cos(30°)
                                            pedestrian_in_path = True
                                            pedestrian_location = actor_location
                                            distance_to_pedestrian = distance
                                            break
                                
                                # Initialize control with default values
                                control = carla.VehicleControl()
                                control.throttle = 0.0
                                control.steer = 0.0
                                control.brake = 0.0
                                control.hand_brake = False
                                control.reverse = False
                                
                                # Get state for RL agent
                                try:
                                    state = self.rl_agent.get_state(self.vehicle, self.world)
                                except Exception as e:
                                    print(f"Error getting RL agent state: {e}")
                                    raise
                                
                                # Get obstacle avoidance predictions
                                try:
                                    obstacles = self.detect_obstacles()
                                except Exception as e:
                                    print(f"Error detecting obstacles: {e}")
                                    raise
                                
                                # Get next waypoint location
                                next_waypoint = self.world.get_map().get_waypoint(vehicle_location).next(5.0)[0]
                                next_waypoint_location = next_waypoint.transform.location
                                
                                # Convert vehicle velocity to scalar speed in km/h
                                vehicle_speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6
                                
                                # Get obstacle avoidance control values
                                try:
                                    throttle, brake, steer = self.obstacle_avoidance.predict_control(
                                        np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z]),
                                        vehicle_speed,
                                        np.array([0, self.vehicle.get_transform().rotation.yaw, 0]),
                                        obstacles,
                                        np.array([next_waypoint_location.x, next_waypoint_location.y, next_waypoint_location.z])
                                    )
                                except Exception as e:
                                    print(f"Error getting obstacle avoidance control: {e}")
                                    raise
                                
                                if pedestrian_in_path and pedestrian_location:
                                    try:
                                        # Get current state for RL agent
                                        state = self.rl_agent.get_state(self.vehicle, self.world)
                                        
                                        # Get obstacle avoidance predictions
                                        throttle, brake, steer = self.obstacle_avoidance.predict_control(
                                            np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z]),
                                            vehicle_speed,
                                            np.array([0, self.vehicle.get_transform().rotation.yaw, 0]),
                                            obstacles,
                                            np.array([next_waypoint_location.x, next_waypoint_location.y, next_waypoint_location.z])
                                        )
                                        
                                        # Get RL agent's action
                                        action = self.rl_agent.select_action(state)
                                        throttle_level = action // 3
                                        steer_level = action % 3
                                        
                                        # Convert RL action to control values
                                        rl_throttle = [0.0, 0.5, 1.0][throttle_level]
                                        rl_steer = [-0.5, 0.0, 0.5][steer_level]
                                        
                                        # Check for available space using ML
                                        current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
                                        left_lane = current_waypoint.get_left_lane()
                                        right_lane = current_waypoint.get_right_lane()
                                        
                                        # Use ML to evaluate available spaces
                                        available_space = []
                                        if left_lane and left_lane.lane_type == carla.LaneType.Driving:
                                            left_lane_location = left_lane.transform.location
                                            left_lane_clear = self.obstacle_avoidance.evaluate_space(
                                                np.array([left_lane_location.x, left_lane_location.y, left_lane_location.z]),
                                                vehicle_speed,
                                                obstacles
                                            )
                                            if left_lane_clear:
                                                available_space.append(('left', left_lane))
                                        
                                        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
                                            right_lane_location = right_lane.transform.location
                                            right_lane_clear = self.obstacle_avoidance.evaluate_space(
                                                np.array([right_lane_location.x, right_lane_location.y, right_lane_location.z]),
                                                vehicle_speed,
                                                obstacles
                                            )
                                            if right_lane_clear:
                                                available_space.append(('right', right_lane))
                                        
                                        # Combine RL decisions with space evaluation
                                        if available_space:
                                            # Choose best space using ML evaluation
                                            best_lane = None
                                            best_score = float('-inf')
                                            
                                            for direction, lane in available_space:
                                                lane_location = lane.transform.location
                                                space_score = self.obstacle_avoidance.score_space(
                                                    np.array([lane_location.x, lane_location.y, lane_location.z]),
                                                    np.array([pedestrian_location.x, pedestrian_location.y, pedestrian_location.z]),
                                                    vehicle_speed
                                                )
                                                if space_score > best_score:
                                                    best_score = space_score
                                                    best_lane = (direction, lane)
                                            
                                            if best_lane:
                                                direction, target_lane = best_lane
                                                target_location = target_lane.transform.location
                                                
                                                # Calculate steering for lane change
                                                target_direction = target_location - vehicle_location
                                                target_direction = target_direction.make_unit_vector()
                                                vehicle_forward = self.vehicle.get_transform().get_forward_vector()
                                                target_cross = vehicle_forward.cross(target_direction)
                                                cross_z = max(-1.0, min(1.0, target_cross.z))
                                                ml_steer = float(math.asin(cross_z))
                                                
                                                # Blend RL and ML controls based on distance to pedestrian
                                                if distance_to_pedestrian < 10.0:
                                                    # Close to pedestrian: More aggressive ML-based navigation
                                                    control.throttle = (rl_throttle * 0.3 + throttle * 0.7)
                                                    control.steer = (rl_steer * 0.3 + ml_steer * 1.5 * 0.7)
                                                    control.brake = brake
                                                    print(f"\nRL+ML: Aggressive navigation through {direction} lane at {distance_to_pedestrian:.1f}m")
                                                else:
                                                    # Further from pedestrian: Balanced RL and ML control
                                                    control.throttle = (rl_throttle * 0.5 + throttle * 0.5)
                                                    control.steer = (rl_steer * 0.5 + ml_steer * 0.5)
                                                    control.brake = brake
                                                    print(f"\nRL+ML: Preparing navigation through {direction} lane")
                                        else:
                                            # No available space: Use RL with obstacle avoidance
                                            if distance_to_pedestrian < 15.0:
                                                # Close to pedestrian: More conservative control
                                                control.throttle = (rl_throttle * 0.2 + throttle * 0.8)
                                                control.steer = (rl_steer * 0.2 + steer * 0.8)
                                                control.brake = brake
                                                print(f"\nRL+ML: Maintaining safe distance at {distance_to_pedestrian:.1f}m")
                                            else:
                                                # Further from pedestrian: More RL influence
                                                control.throttle = (rl_throttle * 0.7 + throttle * 0.3)
                                                control.steer = (rl_steer * 0.7 + steer * 0.3)
                                                control.brake = brake
                                                print(f"\nRL+ML: RL-guided navigation at {distance_to_pedestrian:.1f}m")
                                        
                                        # Update RL agent with experience
                                        reward = self.rl_agent._calculate_ethical_reward(self.vehicle, self.world)
                                        next_state = self.rl_agent.get_state(self.vehicle, self.world)
                                        self.rl_agent.remember(state, action, reward, next_state, False)
                                        loss = self.rl_agent.train()
                                        
                                        print(f"\nRL Agent: Action={action}, Reward={reward:.2f}, Loss={loss:.4f}")
                                    except Exception as e:
                                        print(f"Error in RL+ML navigation: {e}")
                                        # Fall back to obstacle avoidance if RL+ML fails
                                        control.throttle = throttle
                                        control.steer = steer
                                        control.brake = brake
                                else:
                                    # Normal driving conditions - use RL agent
                                    try:
                                        action = self.rl_agent.select_action(state)
                                        throttle_level = action // 3
                                        steer_level = action % 3
                                        
                                        control.throttle = [0.0, 0.5, 1.0][throttle_level]
                                        control.steer = [-0.5, 0.0, 0.5][steer_level]
                                        control.brake = 0.0
                                        
                                        # Calculate reward and update RL agent
                                        reward = self.rl_agent._calculate_ethical_reward(self.vehicle, self.world)
                                        next_state = self.rl_agent.get_state(self.vehicle, self.world)
                                        self.rl_agent.remember(state, action, reward, next_state, False)
                                        loss = self.rl_agent.train()
                                        
                                        print(f"\nRL Agent: Action={action}, Reward={reward:.2f}, Loss={loss:.4f}")
                                    except Exception as e:
                                        print(f"Error in RL agent: {e}")
                                        # Fall back to obstacle avoidance if RL fails
                                        control.throttle = throttle
                                        control.steer = steer
                                        control.brake = brake
                                
                                # Apply control to vehicle
                                try:
                                    self.vehicle.apply_control(control)
                                except Exception as e:
                                    print(f"Error applying control: {e}")
                                    raise
                                
                                # Get next state for RL
                                try:
                                    next_state = self.rl_agent.get_state(self.vehicle, self.world)
                                except Exception as e:
                                    print(f"Error getting next state: {e}")
                                    raise
                                
                                # Calculate reward based on ethical considerations
                                try:
                                    reward = self.rl_agent._calculate_ethical_reward(self.vehicle, self.world)
                                except Exception as e:
                                    print(f"Error calculating reward: {e}")
                                    raise
                                
                                # Additional reward for successful pedestrian avoidance
                                if pedestrian_in_path and distance_to_pedestrian > 5.0:
                                    reward += 0.5  # Reward for finding a clear path around pedestrian
                                
                                # Store experience
                                try:
                                    self.rl_agent.remember(state, action, reward, next_state, False)
                                except Exception as e:
                                    print(f"Error storing experience: {e}")
                                    raise
                                
                                # Train the agent
                                try:
                                    loss = self.rl_agent.train()
                                except Exception as e:
                                    print(f"Error training agent: {e}")
                                    raise
                                
                                # Print vehicle state
                                print(f"\rSpeed: {speed:.2f} km/h, Position: ({vehicle_location.x:.2f}, {vehicle_location.y:.2f}), "
                                      f"Pedestrian in path: {'Yes' if pedestrian_in_path else 'No'}, "
                                      f"Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}, "
                                      f"Steering: {control.steer:.2f}, Epsilon: {self.rl_agent.epsilon:.2f}, "
                                      f"Loss: {loss if loss is not None else 0.0:.4f}", end="")
                    except Exception as e:
                        print(f"Error in waypoint handling: {e}")
                        raise
                    
                except Exception as e:
                    print(f"Error in simulation loop: {e}")
                    self.running = False
                    break
            
        except Exception as e:
            print(f"Error running simulation: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up all actors and resources."""
        try:
            # Destroy all actors
            for actor in self.world.get_actors():
                if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.'):
                    try:
                        actor.destroy()
                    except Exception as e:
                        print(f"Error destroying actor {actor}: {e}")
            
            # Reset world settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            # Reset components
            self.vehicle = None
            self.spectator = None
            self.running = False
            
            print("Cleanup complete")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 