import carla
import time
import math
import numpy as np
import random
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
                x=-10.0 * vehicle_transform.get_forward_vector().x,
                y=-10.0 * vehicle_transform.get_forward_vector().y,
                z=5.0
            )
            
            # Set camera rotation to look at vehicle
            camera_rotation = carla.Rotation(
                pitch=-20.0,
                yaw=vehicle_transform.rotation.yaw
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
                
                # Calculate camera position (behind and above vehicle)
                camera_location = carla.Location(
                    x=vehicle_transform.location.x - 10.0 * math.cos(math.radians(vehicle_transform.rotation.yaw)),
                    y=vehicle_transform.location.y - 10.0 * math.sin(math.radians(vehicle_transform.rotation.yaw)),
                    z=vehicle_transform.location.z + 5.0  # Increased height for better view
                )
                
                # Calculate camera rotation (looking at vehicle)
                camera_rotation = carla.Rotation(
                    pitch=-20.0,  # Slightly steeper angle
                    yaw=vehicle_transform.rotation.yaw,
                    roll=0.0
                )
                
                # Set camera transform
                self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))
                
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
                        
                        # Calculate throttle and brake based on speed difference
                        if traffic_light_state != 'red':  # Only apply speed control if not at red light
                            speed_diff = target_speed - current_velocity
                            if speed_diff > 0:
                                throttle = min(0.3, speed_diff / 2.0)  # Reduced throttle
                                brake = 0.0
                            else:
                                throttle = 0.0
                                brake = min(0.5, -speed_diff / 2.0)
                        
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

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 