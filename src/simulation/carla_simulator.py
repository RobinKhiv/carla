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
                            # Set autopilot
                            vehicle.set_autopilot(True)
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
                    # Check for collisions at spawn point
                    collision = False
                    for actor in self.world.get_actors():
                        if actor.get_location().distance(spawn_point.location) < 5.0:
                            collision = True
                            break
                    
                    if not collision:
                        # Spawn vehicle
                        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                        if self.vehicle is not None:
                            print(f"Vehicle spawned successfully at {spawn_point.location}")
                            # Set vehicle physics
                            self.vehicle.set_simulate_physics(True)
                            # Set vehicle to autopilot mode initially
                            self.vehicle.set_autopilot(False)  # Disable autopilot to allow manual control
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
            # Get vehicle's forward vector
            yaw = math.radians(vehicle_transform.rotation.yaw)
            forward_vector = carla.Location(
                x=math.cos(yaw),
                y=math.sin(yaw),
                z=0
            )
            
            # Calculate camera position (behind and above the vehicle)
            camera_location = vehicle_transform.location + carla.Location(
                x=-10.0 * forward_vector.x,
                y=-10.0 * forward_vector.y,
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
        """Update the camera position to follow the vehicle."""
        try:
            if not self.vehicle or not self.spectator:
                return

            # Get vehicle transform
            vehicle_transform = self.vehicle.get_transform()
            
            # Calculate camera position (behind and above the vehicle)
            # Get vehicle's forward vector
            yaw = math.radians(vehicle_transform.rotation.yaw)
            forward_vector = carla.Location(
                x=math.cos(yaw),
                y=math.sin(yaw),
                z=0
            )
            
            # Calculate camera position (behind and above the vehicle)
            camera_location = vehicle_transform.location + carla.Location(
                x=-10.0 * forward_vector.x,
                y=-10.0 * forward_vector.y,
                z=5.0
            )
            
            # Calculate camera rotation to look at vehicle
            camera_rotation = carla.Rotation(
                pitch=-20.0,
                yaw=vehicle_transform.rotation.yaw
            )
            
            # Set spectator transform
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
            
            # Spawn pedestrians in a line across the road with spacing
            for i in range(5):
                try:
                    # Position pedestrians in a line with spacing
                    pedestrian_location = carla.Location(
                        x=road_location.x + i * 30.0,  # Increased spacing to 30 meters
                        y=road_location.y + 60.0,     # Increased distance ahead to 60 meters
                        z=road_location.z
                    )
                    
                    # Check for collisions before spawning
                    collision = False
                    for actor in existing_actors:
                        if actor.get_location().distance(pedestrian_location) < 30.0:
                            collision = True
                            break
                    
                    if not collision:
                        # Create transform for pedestrian
                        pedestrian_transform = carla.Transform(
                            pedestrian_location,
                            carla.Rotation(yaw=90.0)  # Facing across the road
                        )
                        
                        # Spawn pedestrian with physics disabled
                        walker = self.world.spawn_actor(random.choice(walker_bp), pedestrian_transform)
                        if walker is not None:
                            try:
                                # Disable collisions initially
                                walker.set_simulate_physics(False)
                                
                                # Add to our list
                                self.pedestrians.append(walker)
                                
                                # Add AI controller for the pedestrian
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
                                        
                                        # Make pedestrian walk across the road
                                        target_location = carla.Location(
                                            x=pedestrian_location.x,
                                            y=pedestrian_location.y + 30.0,  # Walk 30 meters across
                                            z=pedestrian_location.z
                                        )
                                        controller.go_to_location(target_location)
                                        controller.set_max_speed(1.4)  # Walking speed
                                        
                                        # Enable physics after controller is attached
                                        walker.set_simulate_physics(True)
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
            
            # Spawn ego vehicle
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
            
            # Spawn traffic
            print("Spawning traffic...")
            self.spawn_pedestrians(10)
            self.spawn_other_vehicles(10)
            
            # Create scenarios
            print("Creating scenarios...")
            self.create_trolley_scenario()
            self.create_hazard_scenario()
            
            # Main simulation loop
            self.running = True
            while self.running:
                try:
                    # Tick the world
                    self.world.tick()
                    
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
                        
                        # Create and apply vehicle control
                        control = carla.VehicleControl(
                            throttle=float(decision.get('throttle', 0.0)),
                            brake=float(decision.get('brake', 0.0)),
                            steer=float(decision.get('steer', 0.0))
                        )
                        
                        # Print control values for debugging
                        print(f"Applying controls - Throttle: {control.throttle}, Brake: {control.brake}, Steer: {control.steer}")
                        
                        # Apply control to vehicle
                        self.vehicle.apply_control(control)
                        
                        # Print vehicle state for debugging
                        velocity = self.vehicle.get_velocity()
                        print(f"Vehicle velocity: {velocity.length()} m/s")
                        
                    except Exception as e:
                        print(f"Error applying controls: {e}")
                        continue
                    
                    # Update camera
                    self.update_camera()
                    
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
        
        # Get the vehicle's location
        vehicle_location = self.vehicle.get_location()
        
        # Find the nearest traffic light
        traffic_light = None
        min_distance = float('inf')
        
        for actor in self.world.get_actors():
            if actor.type_id.startswith('traffic.traffic_light'):
                distance = actor.get_location().distance(vehicle_location)
                if distance < min_distance:
                    min_distance = distance
                    traffic_light = actor
        
        if traffic_light and min_distance < 50.0:  # Only consider traffic lights within 50 meters
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