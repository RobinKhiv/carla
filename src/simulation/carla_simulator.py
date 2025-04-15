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
            self.initialized = False  # Add initialization flag
            
            # Don't initialize here, wait for run() method
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
            
            # Change to Town01 (simple map with straight roads)
            self.client.load_world('Town01')
            self.world = self.client.get_world()
            
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
            
            print("Simulator initialized successfully with Town01 map")
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
                            
                            # Check traffic rules before allowing movement
                            print("Checking traffic rules...")
                            
                            # Check for traffic lights
                            traffic_light_state = self.check_traffic_light()
                            if traffic_light_state == 'red' or traffic_light_state == 'yellow':
                                print(f"Traffic light is {traffic_light_state}, waiting...")
                                # Wait for green light
                                while traffic_light_state != 'green':
                                    self.world.tick()
                                    traffic_light_state = self.check_traffic_light()
                                    time.sleep(0.1)
                                print("Traffic light is green, proceeding...")
                            
                            # Check for stop signs
                            stop_signs = self.world.get_actors().filter('traffic.stop')
                            for stop_sign in stop_signs:
                                if stop_sign.get_location().distance(spawn_point.location) < 10.0:
                                    print("Stop sign detected, waiting...")
                                    time.sleep(2.0)  # Wait at stop sign
                                    print("Proceeding after stop...")
                            
                            # Check for pedestrians
                            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
                            for pedestrian in pedestrians:
                                if pedestrian.get_location().distance(spawn_point.location) < 15.0:
                                    print("Pedestrian detected, waiting...")
                                    time.sleep(2.0)  # Wait for pedestrian
                                    print("Proceeding after pedestrian...")
                            
                            # Set initial control values after traffic checks
                            control = carla.VehicleControl()
                            control.throttle = 0.0  # Start with no throttle
                            control.steer = 0.0
                            control.brake = 0.0
                            self.vehicle.apply_control(control)
                            
                            print("Traffic rules checked, vehicle ready to proceed")
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
            if not self.initialized:  # Only initialize if not already done
                if not self.initialize():
                    raise RuntimeError("Failed to initialize simulator")
                self.initialized = True
            
            # Clean up any existing actors
            self.cleanup()
            
            # Spawn ego vehicle
            print("Spawning ego vehicle...")
            if not self.spawn_vehicle():
                raise RuntimeError("Failed to spawn ego vehicle")
            
            # Set up camera and sensors
            if not self.setup_camera():
                print("Warning: Camera setup failed, continuing without camera")
            
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
                        
                        # Check for traffic lights
                        traffic_light_state = self.check_traffic_light()
                        if traffic_light_state == 'red' or traffic_light_state == 'yellow':
                            print(f"Traffic light is {traffic_light_state}, stopping...")
                            # Apply full brake
                            control = carla.VehicleControl()
                            control.throttle = 0.0
                            control.brake = 1.0
                            control.steer = 0.0
                            self.vehicle.apply_control(control)
                            continue
                        
                        # Get next waypoint with adjusted lookahead distance
                        lookahead_distance = 10.0  # Reduced from 15.0 for more immediate response
                        next_waypoint = current_waypoint.next(lookahead_distance)[0]
                        
                        # Calculate angle to waypoint
                        waypoint_location = next_waypoint.transform.location
                        waypoint_vector = np.array([waypoint_location.x - self.vehicle.get_location().x,
                                                  waypoint_location.y - self.vehicle.get_location().y])
                        vehicle_vector = np.array([np.cos(np.radians(self.vehicle.get_transform().rotation.yaw)),
                                                 np.sin(np.radians(self.vehicle.get_transform().rotation.yaw))])
                        
                        # Normalize vectors and calculate angle
                        waypoint_vector = waypoint_vector / np.linalg.norm(waypoint_vector)
                        vehicle_vector = vehicle_vector / np.linalg.norm(vehicle_vector)
                        angle = np.degrees(np.arccos(np.clip(np.dot(vehicle_vector, waypoint_vector), -1.0, 1.0)))
                        
                        # Determine turn direction
                        cross_product = np.cross(vehicle_vector, waypoint_vector)
                        turn_direction = -1.0 if cross_product < 0 else 1.0
                        
                        # Calculate steering with improved stability
                        steer, current_velocity = self._calculate_steering(self.vehicle.get_transform(), next_waypoint, self.vehicle.get_velocity().length() * 3.6)
                        
                        # Calculate throttle and brake with improved stability
                        throttle, brake = self._calculate_throttle_brake(current_velocity, 5.0, self.vehicle.get_transform(), next_waypoint)
                        
                        # Create and apply vehicle control
                        control = carla.VehicleControl(
                            throttle=float(throttle),
                            brake=float(brake),
                            steer=float(steer)
                        )
                        
                        # Print control values for debugging
                        print(f"Applying controls - Throttle: {control.throttle}, Brake: {control.brake}, Steer: {control.steer}")
                        print(f"Vehicle angle to waypoint: {angle} degrees")
                        print(f"Current velocity: {current_velocity:.2f} km/h")
                        print(f"Traffic light state: {traffic_light_state}")
                        
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

    def check_traffic_light(self):
        """Check the state of the traffic light ahead using CARLA's built-in system."""
        if not self.vehicle:
            return 'unknown'
        
        # Get the vehicle's current waypoint
        vehicle_location = self.vehicle.get_location()
        vehicle_waypoint = self.world.get_map().get_waypoint(vehicle_location)
        
        if not vehicle_waypoint:
            return 'unknown'
        
        # Get the next traffic light
        traffic_light = vehicle_waypoint.get_traffic_light()
        
        if traffic_light:
            # Get the state of the traffic light
            state = traffic_light.get_state()
            if state == carla.TrafficLightState.Green:
                return 'green'
            elif state == carla.TrafficLightState.Yellow:
                return 'yellow'
            elif state == carla.TrafficLightState.Red:
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
            
            # Calculate target speed based on lane position and current velocity
            target_speed = 5.0  # Base target speed in m/s
            
            # Adjust target speed based on lane position
            if left_distance < 0.5 or right_distance < 0.5:
                target_speed *= 0.5  # Slow down when near edges
            elif left_distance < 1.0 or right_distance < 1.0:
                target_speed *= 0.7  # Moderate speed when approaching edges
            
            # Calculate steering correction
            target_position = 0.5  # Center of lane
            position_error = lane_position - target_position
            
            # Apply progressive steering correction
            if abs(position_error) > 0.4:
                steer_correction = -position_error * 0.2  # Stronger correction near edges
            else:
                steer_correction = -position_error * 0.1  # Gentler correction in center
            
            # Smooth steering changes
            if hasattr(self, 'last_steer'):
                self.last_steer = self.last_steer * 0.9 + steer_correction * 0.1
            else:
                self.last_steer = steer_correction
            
            # Apply bounds to steering
            self.last_steer = max(-0.3, min(0.3, self.last_steer))
            
            # Calculate throttle and brake based on velocity difference
            velocity_diff = target_speed - current_velocity
            
            if velocity_diff > 0:
                # Accelerate
                throttle = min(0.3, velocity_diff / 2.0)
                brake = 0.0
            else:
                # Decelerate
                throttle = 0.0
                brake = min(0.3, -velocity_diff / 2.0)
            
            # Check for obstacles
            obstacle_detected = False
            for actor in self.world.get_actors():
                if actor.type_id.startswith('walker.pedestrian'):
                    distance = actor.get_location().distance(vehicle_location)
                    if distance < 10.0:  # Obstacle within 10 meters
                        obstacle_detected = True
                        # Gradually reduce speed when obstacle detected
                        target_speed = min(target_speed, distance * 0.5)
                        break
            
            # Check vehicle stability
            pitch = vehicle_transform.rotation.pitch
            roll = vehicle_transform.rotation.roll
            
            if abs(pitch) > 1.0 or abs(roll) > 1.0:
                # Vehicle is tilted, reduce speed and straighten
                target_speed *= 0.5
                self.last_steer *= 0.5  # Reduce steering when tilted
            
            # Create and apply vehicle control
            control = carla.VehicleControl()
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.steer = float(self.last_steer)
            
            # Print debug information
            print(f"Lane control - Position: {lane_position:.3f}, Error: {position_error:.3f}, Steer: {self.last_steer:.3f}")
            print(f"Target speed: {target_speed:.2f}, Current velocity: {current_velocity:.2f} m/s")
            print(f"Vehicle pitch: {pitch:.1f}°, roll: {roll:.1f}°")
            
            # Apply control to vehicle
            self.vehicle.apply_control(control)
            
        except Exception as e:
            print(f"Error in vehicle control update: {e}")

    def _calculate_steering(self, vehicle_transform, next_waypoint, current_velocity):
        """Calculate steering angle using CARLA's built-in waypoint system."""
        # Get vehicle's current location and rotation
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        # Get the waypoint's transform
        waypoint_transform = next_waypoint.transform
        
        # Calculate the angle between the vehicle's forward vector and the waypoint's forward vector
        vehicle_forward = vehicle_transform.get_forward_vector()
        waypoint_forward = waypoint_transform.get_forward_vector()
        
        # Calculate the angle between the two vectors
        angle = math.degrees(math.acos(
            (vehicle_forward.x * waypoint_forward.x + vehicle_forward.y * waypoint_forward.y) /
            (math.sqrt(vehicle_forward.x**2 + vehicle_forward.y**2) * 
             math.sqrt(waypoint_forward.x**2 + waypoint_forward.y**2))
        ))
        
        # Calculate cross product to determine direction
        cross = vehicle_forward.x * waypoint_forward.y - vehicle_forward.y * waypoint_forward.x
        if cross < 0:
            angle = -angle
        
        # Use CARLA's built-in steering calculation
        # The steering value is normalized between -1.0 (full left) and 1.0 (full right)
        steer = angle / 70.0  # 70 degrees is approximately the maximum steering angle for most vehicles
        
        # Apply bounds to steering
        steer = max(-1.0, min(1.0, steer))
        
        # Apply smoothing to steering
        if hasattr(self, 'last_steer'):
            smoothing_factor = 0.9  # Adjust this value to control steering smoothness
            steer = smoothing_factor * self.last_steer + (1 - smoothing_factor) * steer
        
        self.last_steer = steer
        return steer, current_velocity

    def _calculate_throttle_brake(self, current_velocity, target_velocity, vehicle_transform, next_waypoint):
        """Calculate throttle and brake with improved stability."""
        # Start with a very low target velocity (5 km/h)
        target_velocity = 5.0  # Reduced from 20.0 to 5.0 km/h
        
        # Calculate speed difference
        speed_diff = target_velocity - current_velocity
        
        # Get vehicle's current location and rotation
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        # Calculate vector to next waypoint
        waypoint_vector = carla.Location(
            next_waypoint.transform.location.x - vehicle_location.x,
            next_waypoint.transform.location.y - vehicle_location.y,
            0
        )
        
        # Calculate angle between vehicle's forward vector and waypoint vector
        vehicle_forward = vehicle_transform.get_forward_vector()
        angle = math.degrees(math.acos(
            (vehicle_forward.x * waypoint_vector.x + vehicle_forward.y * waypoint_vector.y) /
            (math.sqrt(vehicle_forward.x**2 + vehicle_forward.y**2) * 
             math.sqrt(waypoint_vector.x**2 + waypoint_vector.y**2))
        ))
        
        # More gradual speed reduction based on angle
        if abs(angle) > 30.0:
            target_velocity *= 0.7  # Reduce speed more for sharp turns
        elif abs(angle) > 15.0:
            target_velocity *= 0.85  # Moderate reduction for turns
        
        # Calculate throttle and brake with more conservative values
        if speed_diff > 0:
            # Very gradual acceleration
            throttle = min(0.2, speed_diff / target_velocity)  # Reduced from 0.4 to 0.2
            brake = 0.0
        else:
            # More aggressive braking to maintain lower speed
            throttle = 0.0
            brake = min(0.3, abs(speed_diff) / target_velocity)
        
        # Apply more smoothing to throttle and brake for smoother transitions
        if hasattr(self, 'last_throttle'):
            smoothing_factor = 0.95  # Increased from 0.9 for even smoother acceleration
            throttle = smoothing_factor * self.last_throttle + (1 - smoothing_factor) * throttle
        if hasattr(self, 'last_brake'):
            smoothing_factor = 0.95  # Increased from 0.9
            brake = smoothing_factor * self.last_brake + (1 - smoothing_factor) * brake
        
        self.last_throttle = throttle
        self.last_brake = brake
        
        return throttle, brake

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 