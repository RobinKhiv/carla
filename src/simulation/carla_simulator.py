import carla
import time
import math
import numpy as np
import random
import cv2
from typing import List, Dict, Any
from ..sensors.sensor_manager import SensorManager
from ..ai.decision_maker import DecisionMaker
from ..ethics.ethical_engine import EthicalEngine
from ..ml.ml_manager import MLManager

class CarlaSimulator:
    def __init__(self, host: str = 'localhost', port: int = 2000):
        """Initialize the CARLA simulator with connection parameters."""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = None
            self.vehicle = None
            self.sensor_manager = None
            self.decision_maker = None
            self.ethical_engine = None
            self.ml_manager = None
            self.spectator = None
            self.running = False
            self.pedestrians = []
            self.other_vehicles = []
            self.walker_controllers = []
            self.traffic_manager = None
            
            # Initialize components
            self.initialize()
        except Exception as e:
            print(f"Error initializing CARLA simulator: {e}")
            raise

    def initialize(self):
        """Initialize the simulation environment and components."""
        try:
            # Get the world
            self.world = self.client.get_world()
            if self.world is None:
                raise RuntimeError("Failed to get CARLA world")
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            
            # Initialize components
            self.sensor_manager = SensorManager(self.world)
            if not self.sensor_manager:
                raise RuntimeError("Failed to initialize sensor manager")
            
            self.decision_maker = DecisionMaker()
            self.ethical_engine = EthicalEngine()
            self.ml_manager = MLManager()
            
            # Set up spectator
            self.spectator = self.world.get_spectator()
            
            # Initialize traffic manager
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.set_synchronous_mode(True)
            
            print("Simulator initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            return False

    def spawn_pedestrians(self, num_pedestrians: int = 20):
        """Spawn pedestrians in the simulation."""
        try:
            # Get pedestrian blueprints
            walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            if not walker_bp:
                print("Warning: No pedestrian blueprints found")
                return
            
            # Get spawn points
            spawn_points = []
            max_attempts = 100
            min_distance = 50.0
            
            # Get all existing actors
            existing_actors = list(self.world.get_actors())
            
            for _ in range(max_attempts):
                try:
                    # Get a random location from navigation mesh
                    spawn_point = carla.Transform()
                    spawn_point.location = self.world.get_random_location_from_navigation()
                    
                    if spawn_point.location is not None:
                        # Check for collisions with existing actors
                        collision = False
                        for actor in existing_actors:
                            if actor.get_location().distance(spawn_point.location) < min_distance:
                                collision = True
                                break
                        
                        if not collision:
                            # Try to spawn a pedestrian
                            walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                            if walker is not None:
                                try:
                                    # Add to our list
                                    self.pedestrians.append(walker)
                                    
                                    # Create and attach controller
                                    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                                    if controller_bp is None:
                                        print("Warning: Failed to find walker controller blueprint")
                                        continue
                                    
                                    controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                                    if controller is not None:
                                        self.walker_controllers.append(controller)
                                        
                                        # Start controller
                                        try:
                                            # Set pedestrian to be non-collidable with other pedestrians
                                            walker.set_simulate_physics(False)
                                            controller.start()
                                            
                                            # Set random destination
                                            destination = self.world.get_random_location_from_navigation()
                                            if destination is not None:
                                                controller.go_to_location(destination)
                                                controller.set_max_speed(1.4)
                                        except Exception as e:
                                            print(f"Warning: Failed to start walker controller: {e}")
                                            if controller in self.walker_controllers:
                                                self.walker_controllers.remove(controller)
                                            controller.destroy()
                                            continue
                                    
                                    spawn_points.append(spawn_point)
                                    if len(spawn_points) >= num_pedestrians:
                                        break
                                except Exception as e:
                                    print(f"Warning: Failed to setup pedestrian controller: {e}")
                                    if walker in self.pedestrians:
                                        self.pedestrians.remove(walker)
                                    walker.destroy()
                                    continue
                except Exception as e:
                    print(f"Warning: Failed to spawn pedestrian at attempt {_}: {e}")
                    continue
            
            print(f"Spawned {len(self.pedestrians)} pedestrians")
        except Exception as e:
            print(f"Error spawning pedestrians: {e}")

    def spawn_other_vehicles(self, num_vehicles: int = 10):
        """Spawn other vehicles in the simulation."""
        try:
            # Get vehicle blueprints
            vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points available for vehicles")
                return
            
            # Configure traffic manager for better behavior
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
                            self.other_vehicles.append(vehicle)
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
            
            print(f"Spawned {len(self.other_vehicles)} vehicles")
        except Exception as e:
            print(f"Error spawning vehicles: {e}")

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
                            # Set vehicle to manual control
                            self.vehicle.set_autopilot(False)
                            # Set initial control values
                            control = carla.VehicleControl()
                            control.throttle = 0.1  # Reduced initial throttle
                            control.steer = 0.0
                            control.brake = 0.0
                            self.vehicle.apply_control(control)
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
                z=vehicle_transform.location.z + 5.0  # Increased height for better view
            )
            
            # Set camera rotation to look at vehicle
            camera_rotation = carla.Rotation(
                pitch=-15.0,  # Looking slightly down at the vehicle
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
                    z=vehicle_transform.location.z + 5.0  # Increased height for better view
                )
                
                # Calculate camera rotation (looking at vehicle)
                camera_rotation = carla.Rotation(
                    pitch=-15.0,  # Looking slightly down at the vehicle
                    yaw=vehicle_transform.rotation.yaw,
                    roll=0.0
                )
                
                # Smoothly interpolate camera position
                current_transform = self.spectator.get_transform()
                target_transform = carla.Transform(camera_location, camera_rotation)
                
                # Use linear interpolation for smoother movement
                smooth_factor = 0.1  # Adjust this value to control smoothness (0.0 to 1.0)
                new_location = carla.Location(
                    x=current_transform.location.x + (target_transform.location.x - current_transform.location.x) * smooth_factor,
                    y=current_transform.location.y + (target_transform.location.y - current_transform.location.y) * smooth_factor,
                    z=current_transform.location.z + (target_transform.location.z - current_transform.location.z) * smooth_factor
                )
                
                # Smoothly interpolate camera rotation
                new_rotation = carla.Rotation(
                    pitch=current_transform.rotation.pitch + (target_transform.rotation.pitch - current_transform.rotation.pitch) * smooth_factor,
                    yaw=current_transform.rotation.yaw + (target_transform.rotation.yaw - current_transform.rotation.yaw) * smooth_factor,
                    roll=current_transform.rotation.roll + (target_transform.rotation.roll - current_transform.rotation.roll) * smooth_factor
                )
                
                # Set camera transform with smoothed values
                self.spectator.set_transform(carla.Transform(new_location, new_rotation))
                
            except Exception as e:
                print(f"Error updating camera: {e}")

    def create_trolley_scenario(self):
        """Create a trolley problem scenario with multiple pedestrians."""
        try:
            # Get pedestrian blueprints
            walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            
            # Create a group of pedestrians crossing the road
            spawn_point = self.vehicle.get_transform()
            road_location = spawn_point.location
            
            # Get all existing actors
            existing_actors = list(self.world.get_actors())
            
            # Spawn pedestrians in a line across the road with increased spacing
            for i in range(5):
                try:
                    # Position pedestrians in a line with increased spacing
                    pedestrian_location = carla.Location(
                        x=road_location.x + i * 50.0,  # Increased spacing to 50 meters
                        y=road_location.y + 100.0,     # Increased distance ahead to 100 meters
                        z=road_location.z
                    )
                    
                    # Check for collisions before spawning with increased radius
                    collision = False
                    for actor in existing_actors:
                        if actor.get_location().distance(pedestrian_location) < 10.0:  # Increased collision radius
                            collision = True
                            break
                    
                    if not collision:
                        # Create transform for pedestrian
                        pedestrian_transform = carla.Transform(
                            pedestrian_location,
                            carla.Rotation(yaw=90.0)  # Facing across the road
                        )
                        
                        # Spawn pedestrian
                        walker = self.world.spawn_actor(random.choice(walker_bp), pedestrian_transform)
                        if walker is not None:
                            try:
                                # Add to our list
                                self.pedestrians.append(walker)
                                
                                # Create and attach controller
                                controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                                if controller_bp is None:
                                    print("Warning: Failed to find walker controller blueprint")
                                    continue
                                
                                controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                                if controller is not None:
                                    self.walker_controllers.append(controller)
                                    
                                    # Start controller
                                    try:
                                        controller.start()
                                        
                                        # Set random destination
                                        destination = carla.Location(
                                            x=pedestrian_location.x,
                                            y=pedestrian_location.y + 30.0,  # Walk 30 meters across
                                            z=pedestrian_location.z
                                        )
                                        controller.go_to_location(destination)
                                        controller.set_max_speed(1.4)  # Walking speed
                                    except Exception as e:
                                        print(f"Warning: Failed to start walker controller: {e}")
                                        if controller in self.walker_controllers:
                                            self.walker_controllers.remove(controller)
                                        controller.destroy()
                                        continue
                            except Exception as e:
                                print(f"Warning: Failed to setup pedestrian controller: {e}")
                                if walker in self.pedestrians:
                                    self.pedestrians.remove(walker)
                                walker.destroy()
                                continue
                except Exception as e:
                    print(f"Warning: Failed to spawn pedestrian {i} in trolley scenario: {e}")
                    continue
            
            print("Created trolley problem scenario with pedestrians crossing the road")
        except Exception as e:
            print(f"Error creating trolley scenario: {e}")

    def create_hazard_scenario(self):
        """Create a hazardous scenario with obstacles and emergency situations."""
        try:
            # Get vehicle blueprints
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            
            # Create a broken-down vehicle scenario
            spawn_point = self.vehicle.get_transform()
            hazard_location = carla.Location(
                x=spawn_point.location.x + 500.0,  # Increased distance to 500 meters
                y=spawn_point.location.y,
                z=spawn_point.location.z
            )
            
            # Check for collisions before spawning
            collision = False
            for actor in self.world.get_actors():
                if actor.get_location().distance(hazard_location) < 50.0:  # Increased collision check distance
                    collision = True
                    break
            
            if not collision:
                # Spawn broken-down vehicle
                hazard_vehicle = self.world.spawn_actor(
                    random.choice(vehicle_bps),
                    carla.Transform(hazard_location, spawn_point.rotation)
                )
                if hazard_vehicle is not None:
                    self.other_vehicles.append(hazard_vehicle)
                    # Set vehicle to be stationary
                    hazard_vehicle.set_simulate_physics(False)
            
            print("Created hazard scenario with broken-down vehicle")
        except Exception as e:
            print(f"Error creating hazard scenario: {e}")

    def run(self):
        """Run the simulation."""
        try:
            if not self.initialize():
                raise RuntimeError("Failed to initialize simulator")
            
            # Clean up any existing actors
            self.cleanup()
            
            # Spawn traffic first
            print("Spawning traffic...")
            self.spawn_pedestrians(10)
            self.spawn_other_vehicles(10)
            
            # Wait a moment for traffic to settle
            for _ in range(10):
                self.world.tick()
            
            # Spawn ego vehicle after traffic
            print("Spawning ego vehicle...")
            if not self.spawn_vehicle():
                raise RuntimeError("Failed to spawn ego vehicle")
            
            # Set up camera and sensors
            if not self.setup_camera():
                print("Warning: Camera setup failed, continuing without camera")
            
            # Create scenarios after vehicle is spawned
            print("Creating scenarios...")
            self.create_trolley_scenario()
            self.create_hazard_scenario()
            
            # Initialize sensor manager if not already done
            if not self.sensor_manager:
                self.sensor_manager = SensorManager(self.world)
                if not self.sensor_manager:
                    raise RuntimeError("Failed to initialize sensor manager")
            
            # Initialize ML manager if not already done
            if not self.ml_manager:
                self.ml_manager = MLManager()
                if not self.ml_manager:
                    raise RuntimeError("Failed to initialize ML manager")
            
            # Initialize sensor data dictionary
            self.sensor_manager.sensor_data = {}
            
            # Setup sensors with proper callback
            def camera_callback(image):
                """Callback for camera sensor data."""
                try:
                    # Convert image to numpy array
                    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (image.height, image.width, 4))
                    array = array[:, :, :3]  # Remove alpha channel
                    
                    # Store the image data
                    if not hasattr(self.sensor_manager, 'sensor_data'):
                        self.sensor_manager.sensor_data = {}
                    self.sensor_manager.sensor_data['camera'] = array
                except Exception as e:
                    print(f"Error in camera callback: {e}")
            
            # Setup camera sensor
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            # Attach camera to vehicle
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            if camera is None:
                raise RuntimeError("Failed to spawn camera sensor")
            
            # Register callback
            camera.listen(camera_callback)
            self.sensor_manager.sensors['camera'] = camera
            
            # Main simulation loop
            self.running = True
            while self.running:
                try:
                    # Tick the world
                    self.world.tick()
                    
                    # Update camera position every frame
                    self.update_camera()
                    
                    # Get sensor data
                    if not self.sensor_manager:
                        print("Warning: Sensor manager not initialized")
                        continue
                    
                    if not hasattr(self.sensor_manager, 'sensor_data'):
                        print("Warning: Sensor data not initialized")
                        continue
                    
                    sensor_data = self.sensor_manager.sensor_data
                    if not sensor_data:
                        print("Warning: No sensor data available")
                        continue
                    
                    if 'camera' not in sensor_data:
                        print("Warning: No camera data available")
                        continue
                    
                    # Process sensor data
                    try:
                        if not self.ml_manager:
                            raise RuntimeError("ML manager not initialized")
                        features = self.ml_manager.process_sensor_data(sensor_data)
                    except Exception as e:
                        print(f"Error processing sensor data: {e}")
                        continue
                    
                    # Make decision
                    try:
                        decision = self.ml_manager.make_decision(features)
                        print(f"Decision: {decision}")  # Debug output
                    except Exception as e:
                        print(f"Error making decision: {e}")
                        continue
                    
                    # Apply controls
                    try:
                        if not self.vehicle:
                            raise RuntimeError("Vehicle not initialized")
                        
                        # Get current waypoint
                        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
                        if current_waypoint is None:
                            print("Warning: Vehicle is not on road")
                            continue
                        
                        # Get next waypoint (look further ahead)
                        next_waypoint = current_waypoint.next(10.0)[0]  # Increased lookahead distance
                        if next_waypoint is None:
                            print("Warning: No next waypoint found")
                            continue
                        
                        # Calculate road curvature with more waypoints
                        road_curvature = 0.0
                        try:
                            # Get multiple waypoints ahead to better estimate curvature
                            next_waypoints = current_waypoint.next(50.0)  # Look much further ahead
                            if len(next_waypoints) > 1:
                                # Calculate average curvature over multiple waypoints
                                total_curvature = 0.0
                                for i in range(len(next_waypoints) - 1):
                                    current_forward = next_waypoints[i].transform.get_forward_vector()
                                    next_forward = next_waypoints[i + 1].transform.get_forward_vector()
                                    turn_vector = current_forward.cross(next_forward)
                                    total_curvature += turn_vector.z
                                road_curvature = total_curvature / (len(next_waypoints) - 1)
                                road_curvature = max(-1.0, min(1.0, road_curvature * 0.8))  # Reduced sensitivity
                        except Exception as e:
                            print(f"Error calculating road curvature: {e}")
                        
                        # Get vehicle transform
                        vehicle_transform = self.vehicle.get_transform()
                        vehicle_location = vehicle_transform.location
                        vehicle_rotation = vehicle_transform.rotation
                        
                        # Calculate direction to next waypoint
                        next_location = next_waypoint.transform.location
                        direction = next_location - vehicle_location
                        direction = direction.make_unit_vector()
                        
                        # Calculate vehicle forward vector
                        vehicle_forward = carla.Vector3D(
                            math.cos(math.radians(vehicle_rotation.yaw)),
                            math.sin(math.radians(vehicle_rotation.yaw)),
                            0
                        )
                        
                        # Calculate angle between vehicle forward vector and direction to next waypoint
                        dot_product = vehicle_forward.dot(direction)
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot_product))))
                        
                        # Calculate cross product to determine turn direction
                        cross_product = vehicle_forward.cross(direction)
                        turn_direction = -1.0 if cross_product.z < 0 else 1.0
                        
                        # Calculate steering based on angle and turn direction
                        max_steer = 0.3  # Reduced from 0.4 to 0.3 for smoother turns
                        angle_factor = min(1.0, angle / 30.0)  # Reduced angle threshold for earlier response
                        base_steer = turn_direction * angle_factor * max_steer
                        
                        # Add road curvature influence
                        base_steer += road_curvature * 0.2  # Add road curvature influence
                        
                        # Apply stronger smoothing to steering
                        if hasattr(self, 'last_steer'):
                            base_steer = self.last_steer * 0.9 + base_steer * 0.1  # Increased smoothing
                        self.last_steer = base_steer
                        
                        # Add lane keeping behavior
                        lane_center_offset = 0.0
                        try:
                            # Get the current lane's center
                            current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
                            if current_waypoint:
                                lane_center = current_waypoint.transform.location
                                # Calculate offset from lane center
                                lane_center_offset = (self.vehicle.get_location().x - lane_center.x) / 3.0  # Normalize by lane width
                                # Add small correction to steering
                                base_steer += lane_center_offset * 0.1  # Reduced lane keeping strength
                        except Exception as e:
                            print(f"Error calculating lane center: {e}")
                        
                        # Print lane keeping info
                        print(f"Lane center offset: {lane_center_offset}")
                        
                        # Check for obstacles with more precise detection
                        obstacle_detected = False
                        min_obstacle_distance = 15.0
                        obstacle_info = []
                        is_pedestrian_on_road = False
                        
                        for actor in self.world.get_actors():
                            # Skip the ego vehicle and non-vehicle/walker actors
                            if actor.id == self.vehicle.id:
                                continue
                            if not (actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.')):
                                continue
                            
                            distance = actor.get_location().distance(vehicle_location)
                            if distance < min_obstacle_distance:
                                # Check if the obstacle is in front of the vehicle
                                obstacle_direction = actor.get_location() - vehicle_location
                                obstacle_direction = obstacle_direction.make_unit_vector()
                                forward_dot = vehicle_forward.dot(obstacle_direction)
                                
                                # For vehicles, use the same criteria as before
                                if actor.type_id.startswith('vehicle.'):
                                    if forward_dot > 0.5:  # More than 60 degrees in front
                                        obstacle_detected = True
                                        obstacle_info.append({
                                            'type': actor.type_id,
                                            'distance': distance,
                                            'forward_dot': forward_dot
                                        })
                                # For pedestrians, be more selective
                                elif actor.type_id.startswith('walker.'):
                                    # Get the waypoint at the pedestrian's location
                                    pedestrian_waypoint = self.world.get_map().get_waypoint(actor.get_location())
                                    if pedestrian_waypoint:
                                        # Only consider pedestrians that are on the road or very close to it
                                        if pedestrian_waypoint.lane_type == carla.LaneType.Driving:
                                            # For pedestrians on the road, be more cautious
                                            if forward_dot > 0.3 and distance < 10.0:  # More than 72 degrees in front and within 10m
                                                obstacle_detected = True
                                                is_pedestrian_on_road = True
                                                obstacle_info.append({
                                                    'type': actor.type_id,
                                                    'distance': distance,
                                                    'forward_dot': forward_dot,
                                                    'lane_type': 'road'
                                                })
                                        else:
                                            # For pedestrians on sidewalks, only stop if they're very close and directly in front
                                            if distance < 3.0 and forward_dot > 0.9:  # Very close and almost directly in front
                                                obstacle_detected = True
                                                obstacle_info.append({
                                                    'type': actor.type_id,
                                                    'distance': distance,
                                                    'forward_dot': forward_dot,
                                                    'lane_type': 'sidewalk'
                                                })
                        
                        # Print obstacle detection info
                        if obstacle_info:
                            print("Obstacles detected:")
                            for info in obstacle_info:
                                print(f"- {info['type']} at {info['distance']:.1f}m, forward_dot: {info['forward_dot']:.2f}")
                                if 'lane_type' in info:
                                    print(f"  Lane type: {info['lane_type']}")
                        else:
                            print("No obstacles detected")
                        
                        # Calculate final steering based on obstacles and lane keeping
                        if obstacle_detected:
                            # When obstacle is detected, maintain lane position but reduce speed
                            steer = base_steer * 0.8  # Reduced steering sensitivity by 20%
                            # Add small correction to stay in lane
                            steer += lane_center_offset * 0.1  # Reduced lane keeping strength
                        else:
                            # Normal driving - use base steering with lane keeping
                            steer = base_steer
                        
                        # Ensure steering stays within bounds
                        steer = max(-0.3, min(0.3, steer))  # Reduced maximum steering angle
                        
                        # Check traffic light state
                        traffic_light_state = self.check_traffic_light()
                        if traffic_light_state == 'red':
                            # Get current velocity
                            current_velocity = self.vehicle.get_velocity().length()
                            
                            # Calculate stopping distance
                            stopping_distance = 5.0  # meters
                            current_distance = self.vehicle.get_location().distance(
                                self.world.get_map().get_waypoint(self.vehicle.get_location()).transform.location
                            )
                            
                            if current_distance <= stopping_distance:
                                # Stop at red light
                                throttle = 0.0
                                brake = 1.0  # Full brake
                                steer = 0.0  # Don't steer while stopped
                                print("Red light detected - stopping")
                            else:
                                # Gradually slow down as approaching red light
                                throttle = 0.0
                                brake = min(0.5, (current_distance - stopping_distance) / 10.0)
                                print("Red light ahead - slowing down")
                        elif traffic_light_state == 'yellow':
                            # Slow down for yellow light
                            speed_factor *= 0.3  # Reduce speed to 30%
                            print("Yellow light detected - slowing down")
                        
                        # Calculate speed based on road curvature and obstacles
                        max_speed = 5.0
                        speed_factor = 1.0 - abs(road_curvature)  # Reduce speed based on curvature
                        
                        # If there's a pedestrian on the road, significantly reduce speed
                        if is_pedestrian_on_road:
                            speed_factor *= 0.3  # Reduce speed to 30% when pedestrian is on road
                        elif obstacle_detected:
                            speed_factor *= 0.7  # Reduce speed to 70% when obstacle is detected
                        
                        # Further reduce speed based on angle to next waypoint
                        if angle > 30.0:  # If turning more than 30 degrees
                            speed_factor *= 0.5  # Reduce speed by half
                        
                        target_speed = max_speed * max(0.2, speed_factor)
                        
                        # Get current velocity
                        current_velocity = self.vehicle.get_velocity().length()
                        
                        # Initialize control variables
                        throttle = 0.0
                        brake = 0.0
                        steer = base_steer
                        
                        # Check if vehicle is stuck (very low velocity but applying throttle)
                        if current_velocity < 0.1 and throttle > 0.1:
                            print("Vehicle appears to be stuck - attempting recovery")
                            # Apply reverse throttle and opposite steering
                            throttle = -0.5  # Increased reverse throttle
                            steer = -steer * 0.8  # Increased steering magnitude
                            brake = 0.0
                            # Wait for a moment to allow recovery
                            time.sleep(0.5)
                        else:
                            # Normal speed control
                            speed_diff = target_speed - current_velocity
                            if speed_diff > 0:
                                throttle = min(0.3, speed_diff / 2.0)  # Reduced throttle
                                brake = 0.0
                            else:
                                throttle = 0.0
                                brake = min(0.5, -speed_diff / 2.0)
                        
                        # Check vehicle orientation
                        vehicle_transform = self.vehicle.get_transform()
                        pitch = vehicle_transform.rotation.pitch
                        roll = vehicle_transform.rotation.roll
                        
                        # If vehicle is tilted too much, try to recover
                        if abs(pitch) > 1.0 or abs(roll) > 1.0:
                            print(f"Vehicle tilted - pitch: {pitch}, roll: {roll}")
                            # More aggressive recovery for tilted vehicle
                            throttle = 0.0  # Stop applying throttle
                            brake = 0.3  # Apply moderate brake
                            steer = 0.0  # Reset steering
                            # Wait for a moment to allow stabilization
                            time.sleep(0.5)
                        
                        # Create and apply vehicle control
                        control = carla.VehicleControl(
                            throttle=float(throttle),
                            brake=float(brake),
                            steer=float(steer)  # Use calculated steering
                        )
                        
                        # Print control values for debugging
                        print(f"Applying controls - Throttle: {control.throttle}, Brake: {control.brake}, Steer: {control.steer}")
                        print(f"Vehicle angle to waypoint: {angle} degrees")
                        print(f"Obstacle detected: {obstacle_detected}")
                        print(f"Current velocity: {current_velocity} m/s")
                        print(f"Vehicle pitch: {pitch}, roll: {roll}")
                        
                        # Apply control to vehicle
                        self.vehicle.apply_control(control)
                        
                        # Print vehicle state for debugging
                        velocity = self.vehicle.get_velocity()
                        print(f"Vehicle velocity: {velocity.length()} m/s")
                        
                        # Print vehicle transform for debugging
                        transform = self.vehicle.get_transform()
                        print(f"Vehicle position: {transform.location}")
                        print(f"Vehicle rotation: {transform.rotation}")
                        
                        # Print waypoint information
                        print(f"Current waypoint: {current_waypoint.transform.location}")
                        print(f"Next waypoint: {next_waypoint.transform.location}")
                        
                    except Exception as e:
                        print(f"Error applying controls: {e}")
                        continue
                    
                except Exception as e:
                    print(f"Error in simulation loop: {e}")
                    self.running = False
                    break
            
        except Exception as e:
            print(f"Error running simulation: {e}")
        finally:
            self.cleanup()

    def check_traffic_light(self) -> str:
        """Check the state of the traffic light ahead."""
        if not self.vehicle:
            return 'unknown'
        
        # Get the vehicle's location and transform
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        # Get the vehicle's forward vector
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        # Find the nearest traffic light
        traffic_light = None
        min_distance = float('inf')
        
        for actor in self.world.get_actors():
            if actor.type_id.startswith('traffic.traffic_light'):
                # Get traffic light location
                light_location = actor.get_location()
                
                # Calculate vector from vehicle to traffic light
                light_direction = light_location - vehicle_location
                light_direction = light_direction.make_unit_vector()
                
                # Check if traffic light is in front of the vehicle
                forward_dot = vehicle_forward.dot(light_direction)
                if forward_dot > 0.5:  # Traffic light is in front of vehicle
                    distance = vehicle_location.distance(light_location)
                    if distance < min_distance:
                        min_distance = distance
                        traffic_light = actor
        
        if traffic_light and min_distance < 50.0:  # Only consider traffic lights within 50 meters
            # Get the waypoint at the traffic light
            light_waypoint = self.world.get_map().get_waypoint(traffic_light.get_location())
            if light_waypoint:
                # Calculate stopping distance (5 meters before the traffic light)
                stopping_distance = 5.0
                if min_distance <= stopping_distance:
                    state = traffic_light.get_state()
                    if state == carla.TrafficLightState.Green:
                        return 'green'
                    elif state == carla.TrafficLightState.Yellow:
                        return 'yellow'
                    else:
                        return 'red'
        
        return 'unknown'

    def apply_decision(self, decision: Dict[str, Any]):
        """Apply the decision to the vehicle."""
        if not self.vehicle:
            return

        # Get controls from the decision dictionary
        controls = decision.get('controls', {})
        throttle = controls.get('throttle', 0.0)
        brake = controls.get('brake', 0.0)
        steer = controls.get('steer', 0.0)

        # Create and apply vehicle control
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steer)
        
        self.vehicle.apply_control(control)

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
            
            # Clear lists
            self.pedestrians.clear()
            self.other_vehicles.clear()
            self.walker_controllers.clear()
            
            # Reset components
            self.vehicle = None
            self.sensor_manager = None
            self.decision_maker = None
            self.ethical_engine = None
            self.ml_manager = None
            self.spectator = None
            self.running = False
            
            print("Cleanup complete")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def setup_sensors(self):
        """Set up additional sensors for lane detection and keeping."""
        try:
            if not self.vehicle:
                print("Warning: Cannot setup sensors - no vehicle available")
                return False

            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            
            # Add lane detection camera
            lane_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
            lane_camera_bp.set_attribute('image_size_x', '800')
            lane_camera_bp.set_attribute('image_size_y', '600')
            lane_camera_bp.set_attribute('fov', '90')
            
            # Position camera to look at the road
            lane_camera_transform = carla.Transform(
                carla.Location(x=1.5, z=2.4),  # Mounted on the front of the vehicle
                carla.Rotation(pitch=-15.0)     # Looking slightly down at the road
            )
            
            # Spawn and attach the camera
            self.lane_camera = self.world.spawn_actor(
                lane_camera_bp,
                lane_camera_transform,
                attach_to=self.vehicle
            )
            
            # Add callback for lane detection
            self.lane_camera.listen(self.process_lane_detection)
            
            # Add LIDAR for precise distance measurements
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '50.0')  # 50 meters range
            lidar_bp.set_attribute('points_per_second', '100000')
            lidar_bp.set_attribute('rotation_frequency', '10')
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('upper_fov', '10.0')
            lidar_bp.set_attribute('lower_fov', '-30.0')
            
            # Position LIDAR on top of the vehicle
            lidar_transform = carla.Transform(
                carla.Location(x=0.0, z=2.5),
                carla.Rotation(pitch=0.0)
            )
            
            # Spawn and attach the LIDAR
            self.lidar = self.world.spawn_actor(
                lidar_bp,
                lidar_transform,
                attach_to=self.vehicle
            )
            
            # Add callback for LIDAR data
            self.lidar.listen(self.process_lidar_data)
            
            print("Additional sensors setup complete")
            return True
            
        except Exception as e:
            print(f"Error setting up sensors: {e}")
            return False

    def process_lane_detection(self, image):
        """Process lane detection camera data."""
        try:
            # Convert image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            
            # Process lane markings
            lane_markings = self.detect_lane_markings(array)
            
            # Update lane keeping parameters based on detected markings
            if lane_markings:
                self.update_lane_keeping(lane_markings)
                
        except Exception as e:
            print(f"Error processing lane detection: {e}")

    def process_lidar_data(self, data):
        """Process LIDAR data for precise distance measurements."""
        try:
            # Convert LIDAR data to numpy array
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            
            # Process points to detect lane boundaries and obstacles
            lane_boundaries = self.detect_lane_boundaries(points)
            
            # Update vehicle control based on detected boundaries
            if lane_boundaries:
                self.update_vehicle_control(lane_boundaries)
                
        except Exception as e:
            print(f"Error processing LIDAR data: {e}")

    def detect_lane_markings(self, image):
        """Detect lane markings in the camera image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Apply Hough transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
            
            if lines is not None:
                left_lines = []
                right_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate line parameters
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        
                        # Classify as left or right lane
                        if slope < 0:
                            left_lines.append((slope, intercept))
                        else:
                            right_lines.append((slope, intercept))
                
                # Calculate average lane positions
                left_lane = np.mean([-intercept/slope for slope, intercept in left_lines]) if left_lines else None
                right_lane = np.mean([-intercept/slope for slope, intercept in right_lines]) if right_lines else None
                
                return {'left': left_lane, 'right': right_lane}
            
            return None
            
        except Exception as e:
            print(f"Error in lane marking detection: {e}")
            return None

    def detect_lane_boundaries(self, points):
        """Detect lane boundaries from LIDAR data."""
        try:
            # Filter points to road surface
            road_points = points[points[:, 2] > -0.5]  # Points above -0.5m in height
            
            if len(road_points) > 0:
                # Calculate lane boundaries using point cloud
                left_boundary = np.min(road_points[:, 0])
                right_boundary = np.max(road_points[:, 0])
                
                return {'left': left_boundary, 'right': right_boundary}
            
            return None
            
        except Exception as e:
            print(f"Error in lane boundary detection: {e}")
            return None

    def update_lane_keeping(self, lane_markings):
        """Update lane keeping parameters based on detected lane markings."""
        try:
            if not lane_markings:
                return

            # Calculate lane center and offset
            left_lane = lane_markings.get('left', None)
            right_lane = lane_markings.get('right', None)
            
            if left_lane and right_lane:
                # Calculate lane center
                lane_center = (left_lane + right_lane) / 2.0
                
                # Get vehicle position relative to lane center
                vehicle_location = self.vehicle.get_location()
                vehicle_transform = self.vehicle.get_transform()
                
                # Calculate offset from lane center
                lane_offset = vehicle_location.x - lane_center
                
                # Calculate angle to lane center
                lane_angle = math.atan2(lane_center - vehicle_location.x, 
                                      vehicle_location.y - vehicle_location.y)
                vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
                angle_diff = lane_angle - vehicle_yaw
                
                # Calculate steering correction
                offset_correction = -lane_offset * 0.1  # Proportional gain
                angle_correction = -angle_diff * 0.2    # Proportional gain
                
                # Combine corrections with smoothing
                if hasattr(self, 'last_steer'):
                    self.last_steer = self.last_steer * 0.8 + (offset_correction + angle_correction) * 0.2
                else:
                    self.last_steer = offset_correction + angle_correction
                
                # Apply bounds to steering
                self.last_steer = max(-0.3, min(0.3, self.last_steer))
                
                # Print debug information
                print(f"Lane keeping - Offset: {lane_offset:.3f}m, Angle: {math.degrees(angle_diff):.1f}°, Steer: {self.last_steer:.3f}")
                
        except Exception as e:
            print(f"Error in lane keeping update: {e}")

    def update_vehicle_control(self, lane_boundaries):
        """Update vehicle control based on detected lane boundaries."""
        try:
            if not lane_boundaries:
                return

            # Get current vehicle state
            vehicle_location = self.vehicle.get_location()
            vehicle_transform = self.vehicle.get_transform()
            current_velocity = self.vehicle.get_velocity().length()
            
            # Calculate distance to lane boundaries
            left_distance = abs(vehicle_location.x - lane_boundaries['left'])
            right_distance = abs(vehicle_location.x - lane_boundaries['right'])
            
            # Calculate lane width
            lane_width = abs(lane_boundaries['right'] - lane_boundaries['left'])
            
            # Calculate normalized position in lane (0 = left edge, 1 = right edge)
            lane_position = (vehicle_location.x - lane_boundaries['left']) / lane_width
            
            # Calculate target speed based on lane position
            # Slow down when too close to either edge
            speed_factor = 1.0
            if left_distance < 0.5 or right_distance < 0.5:
                speed_factor = 0.5
            elif left_distance < 1.0 or right_distance < 1.0:
                speed_factor = 0.7
            
            # Calculate steering correction
            target_position = 0.5  # Center of lane
            position_error = lane_position - target_position
            
            # Apply stronger correction when near lane edges
            if abs(position_error) > 0.4:
                steer_correction = -position_error * 0.3
            else:
                steer_correction = -position_error * 0.1
            
            # Smooth steering changes
            if hasattr(self, 'last_steer'):
                self.last_steer = self.last_steer * 0.9 + steer_correction * 0.1
            else:
                self.last_steer = steer_correction
            
            # Apply bounds to steering
            self.last_steer = max(-0.3, min(0.3, self.last_steer))
            
            # Print debug information
            print(f"Lane control - Position: {lane_position:.3f}, Error: {position_error:.3f}, Steer: {self.last_steer:.3f}")
            print(f"Speed factor: {speed_factor:.2f}, Current velocity: {current_velocity:.2f} m/s")
            
        except Exception as e:
            print(f"Error in vehicle control update: {e}")

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 