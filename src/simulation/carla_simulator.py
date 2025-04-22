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
            
            # Camera smoothing parameters
            self.camera_smoothing_factor = 0.03  # Lower values = smoother movement
            self.last_camera_transform = None
            
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
            
            # Try to load Town10 (downtown map)
            try:
                print("Loading Town10 map...")
                self.client.load_world('Town10HD_Opt')
            except Exception as e:
                print(f"Failed to load Town10: {e}")
                raise RuntimeError("Failed to load Town10 map")
            
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
            
            # Initialize obstacle avoidance model
            print("Initializing obstacle avoidance model...")
            self.obstacle_avoidance = ObstacleAvoidance(self.world)
            
            print("Simulator initialized successfully")
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            print("Please ensure that:")
            print("1. CARLA server is running")
            print("2. CARLA server is accessible at localhost:2000")
            print("3. You have the correct version of CARLA installed")
            self.initialized = False
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
        """Update camera position and orientation with smooth movement"""
        if self.vehicle and self.spectator:
            try:
                # Get vehicle transform
                vehicle_transform = self.vehicle.get_transform()
                
                # Calculate target camera position (behind and above the vehicle)
                target_location = carla.Location(
                    x=vehicle_transform.location.x - 10.0 * math.cos(math.radians(vehicle_transform.rotation.yaw)),
                    y=vehicle_transform.location.y - 10.0 * math.sin(math.radians(vehicle_transform.rotation.yaw)),
                    z=vehicle_transform.location.z + 5.0
                )
                
                # Calculate target camera rotation (looking at vehicle)
                target_rotation = carla.Rotation(
                    pitch=-15.0,
                    yaw=vehicle_transform.rotation.yaw,
                    roll=0.0
                )
                
                # Create target transform
                target_transform = carla.Transform(target_location, target_rotation)
                
                # If this is the first update, set the camera directly
                if self.last_camera_transform is None:
                    self.spectator.set_transform(target_transform)
                    self.last_camera_transform = target_transform
                    return
                
                # Interpolate between current and target position
                current_location = self.last_camera_transform.location
                current_rotation = self.last_camera_transform.rotation
                
                # Smooth position interpolation
                new_location = carla.Location(
                    x=current_location.x + (target_location.x - current_location.x) * self.camera_smoothing_factor,
                    y=current_location.y + (target_location.y - current_location.y) * self.camera_smoothing_factor,
                    z=current_location.z + (target_location.z - current_location.z) * self.camera_smoothing_factor
                )
                
                # Smooth rotation interpolation
                new_rotation = carla.Rotation(
                    pitch=current_rotation.pitch + (target_rotation.pitch - current_rotation.pitch) * self.camera_smoothing_factor,
                    yaw=current_rotation.yaw + (target_rotation.yaw - current_rotation.yaw) * self.camera_smoothing_factor,
                    roll=current_rotation.roll + (target_rotation.roll - current_rotation.roll) * self.camera_smoothing_factor
                )
                
                # Set new camera transform
                new_transform = carla.Transform(new_location, new_rotation)
                self.spectator.set_transform(new_transform)
                self.last_camera_transform = new_transform
                
            except Exception as e:
                print(f"Error updating camera: {e}")

    def spawn_traffic(self, num_vehicles: int = 10, num_pedestrians: int = 50):
        """Spawn traffic and create specific scenarios for testing ethical decision making."""
        try:
            # Get vehicle blueprints
            vehicle_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points available for vehicles")
                return
            
            # Configure traffic manager
            self.traffic_manager.set_global_distance_to_leading_vehicle(5.0)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(0)
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
            
            # Spawn vehicles with proper lane spacing
            vehicles_spawned = 0
            used_lanes = set()  # Track which lanes have vehicles
            
            for i in range(min(num_vehicles, len(spawn_points))):
                try:
                    spawn_point = spawn_points[i]
                    
                    # Get the waypoint at this spawn point
                    waypoint = self.world.get_map().get_waypoint(spawn_point.location)
                    if not waypoint:
                        continue
                    
                    # Check if this lane is already occupied
                    lane_id = (waypoint.road_id, waypoint.lane_id)
                    if lane_id in used_lanes:
                        print(f"Skipping spawn point {i} - lane already occupied")
                        continue
                    
                    # Check for collisions at spawn point
                    collision = False
                    for actor in self.world.get_actors():
                        if actor.get_location().distance(spawn_point.location) < 10.0:
                            collision = True
                            break
                    
                    if not collision:
                        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                        if vehicle is not None:
                            # Set autopilot with traffic manager
                            vehicle.set_autopilot(True, self.traffic_manager.get_port())
                            
                            # Configure traffic manager settings
                            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 30.0)
                            self.traffic_manager.auto_lane_change(vehicle, False)
                            self.traffic_manager.distance_to_leading_vehicle(vehicle, 5.0)
                            
                            # Mark this lane as used
                            used_lanes.add(lane_id)
                            vehicles_spawned += 1
                            print(f"Spawned vehicle {vehicles_spawned} in lane {lane_id}")
                except Exception as e:
                    print(f"Warning: Failed to spawn vehicle at point {i}: {e}")
                    continue
            
            # Create pedestrian scenarios in downtown areas
            pedestrians_spawned = 0
            try:
                # Get pedestrian blueprints
                walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
                print(f"Found {len(walker_bp)} pedestrian blueprints")
                
                # Get all waypoints in the map
                waypoints = self.world.get_map().generate_waypoints(1.0)
                print(f"Total waypoints found: {len(waypoints)}")
                
                # Filter waypoints that are in downtown areas (not highways)
                downtown_waypoints = []
                for wp in waypoints:
                    # Look for waypoints in downtown areas (narrower roads, intersections)
                    if (wp.lane_type == carla.LaneType.Driving and 
                        not wp.is_junction and 
                        len(wp.next(1.0)) <= 2):  # Not a highway
                        downtown_waypoints.append(wp)
                
                if not downtown_waypoints:
                    print("Warning: No downtown waypoints found")
                    return
                
                print(f"Found {len(downtown_waypoints)} downtown waypoints")
                
                # Function to try spawning a pedestrian at a location
                def try_spawn_pedestrian(spawn_point, waypoint):
                    nonlocal pedestrians_spawned
                    try:
                        # Check for collisions with a larger radius
                        if any(actor.get_location().distance(spawn_point.location) < 3.0 
                              for actor in self.world.get_actors()):
                            return False
                        
                        # Check if the spawn point is on a valid surface
                        if not self.world.get_map().get_waypoint(spawn_point.location):
                            return False
                        
                        walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                        if walker is not None:
                            # Find a destination waypoint
                            next_waypoints = waypoint.next(10.0)
                            if next_waypoints:
                                destination = random.choice(next_waypoints).transform.location
                                destination.z = spawn_point.location.z
                                walker.set_location(destination)
                                pedestrians_spawned += 1
                                print(f"Successfully spawned pedestrian {pedestrians_spawned} at {spawn_point.location}")
                                return True
                    except Exception as e:
                        print(f"Failed to spawn pedestrian: {e}")
                    return False
                
                # Spawn pedestrians in downtown areas
                num_sections = 10  # Increased from 5 to 10 sections
                pedestrians_per_section = 2  # 2 pedestrians per section
                
                for section in range(num_sections):
                    print(f"\nAttempting to spawn pedestrians in section {section + 1}")
                    waypoint = random.choice(downtown_waypoints)
                    spawn_location = waypoint.transform.location
                    
                    # Calculate road direction and perpendicular
                    road_direction = waypoint.transform.get_forward_vector()
                    perpendicular = carla.Vector3D(-road_direction.y, road_direction.x, 0)
                    
                    # Try different spawn points in this section
                    for i in range(pedestrians_per_section):
                        # Try different offsets, but ensure we don't block both sides
                        # Alternate between left and right side of the road
                        side = 1 if i % 2 == 0 else -1
                        offset = side * 2.0  # Fixed 2m offset from center
                        
                        spawn_point = carla.Transform()
                        spawn_point.location = spawn_location + perpendicular * offset + road_direction * (i * 5.0)
                        spawn_point.location.z += 0.5
                        
                        # Try up to 3 different positions if spawning fails
                        for attempt in range(3):
                            if try_spawn_pedestrian(spawn_point, waypoint):
                                print(f"Spawned pedestrian on {'right' if side > 0 else 'left'} side of road")
                                break
                            else:
                                # Try a slightly different position
                                spawn_point.location += carla.Location(0, 0, 0.1)  # Move up slightly
                    
                    # Move to next waypoint for next section
                    next_waypoints = waypoint.next(30.0)  # Reduced from 50m to 30m between sections
                    if next_waypoints:
                        waypoint = random.choice(next_waypoints)
                
                print(f"\nTotal pedestrians spawned: {pedestrians_spawned}")
                print(f"Spawned {vehicles_spawned} vehicles and {pedestrians_spawned} pedestrians in downtown")
                
            except Exception as e:
                print(f"Error spawning pedestrians: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error spawning traffic: {e}")
            import traceback
            traceback.print_exc()

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
            self.spawn_traffic(3, 20)  # Spawn 10 vehicles and 50 pedestrians
            
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
            
            # Configure traffic manager to prevent actor destruction
            self.traffic_manager.set_global_distance_to_leading_vehicle(5.0)  # Increase following distance
            self.traffic_manager.set_respawn_dormant_vehicles(False)  # Disable respawn of dormant vehicles
            self.traffic_manager.set_hybrid_physics_mode(True)  # Enable hybrid physics for better control
            
            # Main simulation loop
            self.running = True
            last_vehicle_check = time.time()
            
            while self.running:
                try:
                    # Tick the world
                    self.world.tick()
                    
                    # Update camera position
                    self.update_camera()
                    
                    # Check vehicle status periodically
                    current_time = time.time()
                    if current_time - last_vehicle_check > 1.0:  # Check every second
                        if not self.vehicle.is_alive:
                            print("Warning: Vehicle was destroyed, attempting to respawn...")
                            if not self.spawn_vehicle():
                                print("Error: Failed to respawn vehicle")
                                self.running = False
                                break
                        last_vehicle_check = current_time
                    
                    # Get vehicle state
                    try:
                        vehicle_location = self.vehicle.get_location()
                        vehicle_velocity = self.vehicle.get_velocity()
                        # Calculate speed in km/h using CARLA's velocity vector
                        speed = 3.6 * math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
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
                                    print(f"\n[DEBUG] RL State: {state}")
                                except Exception as e:
                                    print(f"Error getting RL agent state: {e}")
                                    raise
                                
                                # Get obstacle avoidance predictions
                                try:
                                    obstacles = self.detect_obstacles()
                                    pedestrians = [obs for obs in obstacles if obs[2] == 'pedestrian']
                                    
                                    if pedestrians:
                                        print(f"\n[PEDESTRIAN DETECTED] Found {len(pedestrians)} pedestrians:")
                                        for i, (loc, dist, _) in enumerate(pedestrians):
                                            print(f"  Pedestrian {i+1}: {dist:.1f}m away")
                                    
                                    # Get ML control predictions
                                    throttle, brake, steer = self.obstacle_avoidance.predict_control(
                                        np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z]),
                                        speed,
                                        np.array([0, self.vehicle.get_transform().rotation.yaw, 0]),
                                        obstacles,
                                        np.array([next_waypoint.transform.location.x, next_waypoint.transform.location.y, next_waypoint.transform.location.z])
                                    )
                                    
                                    if pedestrians:
                                        print(f"[MODEL DECISION] Using ML model for pedestrian avoidance:")
                                        print(f"  - Throttle: {throttle:.2f}")
                                        print(f"  - Brake: {brake:.2f}")
                                        print(f"  - Steer: {steer:.2f}")
                                except Exception as e:
                                    print(f"Error in ML prediction: {e}")
                                
                                # Get RL agent state and action
                                try:
                                    state = self.rl_agent.get_state(self.vehicle, self.world)
                                    action = self.rl_agent.select_action(state)
                                    
                                    if pedestrians:
                                        print(f"[RL DECISION] High-level action: {action_names.get(action, 'UNKNOWN')}")
                                    
                                    # Train RL agent
                                    next_state = self.rl_agent.get_state(self.vehicle, self.world)
                                    reward = self.rl_agent._calculate_ethical_reward(self.vehicle, self.world)
                                    self.rl_agent.remember(state, action, reward, next_state, False)
                                    loss = self.rl_agent.train()
                                except Exception as e:
                                    print(f"Error in RL agent: {e}")
                                
                                # Combine ML and RL controls
                                try:
                                    # Use ML for obstacle avoidance and RL for high-level decisions
                                    if len(obstacles) > 0:
                                        control.throttle = throttle
                                        control.brake = brake
                                        control.steer = steer
                                        if pedestrians:
                                            print("[CONTROL] Using ML-based obstacle avoidance controls")
                                    else:
                                        # Use RL action for normal driving
                                        if action == 0:  # Accelerate
                                            control.throttle = 0.5
                                            control.brake = 0.0
                                        elif action == 1:  # Brake
                                            control.throttle = 0.0
                                            control.brake = 0.5
                                        elif action == 2:  # Steer left
                                            control.steer = -0.5
                                        elif action == 3:  # Steer right
                                            control.steer = 0.5
                                        if pedestrians:
                                            print("[CONTROL] Using RL-based normal driving controls")
                                except Exception as e:
                                    print(f"Error combining controls: {e}")
                                
                                # Check traffic light state
                                try:
                                    # Get the traffic light affecting the vehicle
                                    traffic_light = None
                                    for actor in self.world.get_actors().filter('traffic.traffic_light'):
                                        if actor.get_location().distance(vehicle_location) < 30.0:  # Check within 30 meters
                                            traffic_light = actor
                                            break
                                    
                                    if traffic_light:
                                        light_state = traffic_light.get_state()
                                        
                                        # Handle traffic light states
                                        if light_state == carla.TrafficLightState.Red:
                                            control.throttle = 0.0
                                            control.brake = 0.5  # Reduced from 1.0 to prevent sudden stops
                                        elif light_state == carla.TrafficLightState.Yellow:
                                            control.throttle = 0.0
                                            control.brake = 0.3  # Reduced from 0.5
                                        elif light_state == carla.TrafficLightState.Green:
                                            # Resume normal control
                                            if speed < 5.0:
                                                control.throttle = 0.3  # Reduced from 0.5 for smoother acceleration
                                                control.brake = 0.0
                                except Exception as e:
                                    print(f"Error checking traffic light: {e}")
                                
                                # Ensure minimum movement
                                try:
                                    if speed < 5.0 and control.throttle < 0.3:
                                        control.throttle = 0.3
                                        control.brake = 0.0
                                except Exception as e:
                                    print(f"Error in speed control: {e}")
                                
                                # Print vehicle state with safe formatting
                                try:
                                    print(f"\rSpeed: {speed:.2f} km/h, Position: ({vehicle_location.x:.2f}, {vehicle_location.y:.2f}), "
                                          f"Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}, "
                                          f"Steering: {control.steer:.2f}", end="")
                                except Exception as e:
                                    print(f"Error printing vehicle state: {e}")
                                
                                # Apply control to vehicle
                                try:
                                    self.vehicle.apply_control(control)
                                except Exception as e:
                                    print(f"Error applying control: {e}")
                                    raise
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