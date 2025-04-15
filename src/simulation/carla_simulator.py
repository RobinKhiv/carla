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
            self.decision_maker = DecisionMaker()
            self.ethical_engine = EthicalEngine()
            self.ml_manager = MLManager()
            
            # Set up spectator
            self.spectator = self.world.get_spectator()
            
            # Initialize traffic manager
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.set_synchronous_mode(True)
            
            return True
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            return False

    def spawn_pedestrians(self, num_pedestrians: int = 20):
        """Spawn pedestrians in the simulation."""
        try:
            # Get pedestrian blueprints
            walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            
            # Get spawn points
            spawn_points = []
            max_attempts = 100  # Maximum attempts to find valid spawn points
            min_distance = 50.0  # Increased minimum distance between pedestrians
            
            for _ in range(max_attempts):
                spawn_point = carla.Transform()
                spawn_point.location = self.world.get_random_location_from_navigation()
                
                # Check if the spawn point is valid (not colliding)
                if spawn_point.location is not None:
                    # Check for collisions at the spawn point
                    collision = False
                    for actor in self.world.get_actors():
                        if actor.get_location().distance(spawn_point.location) < min_distance:
                            collision = True
                            break
                    
                    if not collision:
                        spawn_points.append(spawn_point)
                        if len(spawn_points) >= num_pedestrians:
                            break
            
            # Spawn pedestrians
            for spawn_point in spawn_points:
                walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                if walker is not None:
                    self.pedestrians.append(walker)
                    # Add AI controller for the pedestrian
                    controller = self.world.spawn_actor(
                        self.world.get_blueprint_library().find('controller.ai.walker'),
                        carla.Transform(),
                        walker
                    )
                    if controller is not None:
                        self.walker_controllers.append(controller)
                        controller.start()
                        # Make pedestrian walk
                        controller.go_to_location(self.world.get_random_location_from_navigation())
                        controller.set_max_speed(1.4)  # Walking speed
            
            print(f"Spawned {len(self.pedestrians)} pedestrians")
        except Exception as e:
            print(f"Error spawning pedestrians: {e}")

    def spawn_other_vehicles(self, num_vehicles: int = 20):
        """Spawn other vehicles in the simulation."""
        try:
            # Get vehicle blueprints
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_points = random.sample(spawn_points, min(num_vehicles, len(spawn_points)))
            
            # Spawn vehicles
            for spawn_point in spawn_points:
                vehicle = self.world.spawn_actor(random.choice(vehicle_bps), spawn_point)
                self.other_vehicles.append(vehicle)
                self.traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 50))
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, random.randint(0, 30))
                self.traffic_manager.auto_lane_change(vehicle, True)
            
            print(f"Spawned {len(self.other_vehicles)} other vehicles")
        except Exception as e:
            print(f"Error spawning other vehicles: {e}")

    def spawn_vehicle(self, spawn_point_index: int = 0) -> bool:
        """Spawn a vehicle at the specified spawn point."""
        try:
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points available")
                return False

            # Choose spawn point
            spawn_point = spawn_points[spawn_point_index]
            
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if vehicle_bp is None:
                vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Vehicle spawned at: {spawn_point.location}")

            # Attach sensors
            self.sensor_manager.attach_sensors(self.vehicle)
            
            return True
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return False

    def setup_camera(self):
        """Set up the spectator camera to follow the vehicle."""
        if not self.vehicle:
            return

        # Initial camera setup
        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        
        # Calculate camera position
        offset = carla.Location(
            x=-15 * math.cos(yaw),
            y=15 * math.sin(yaw),
            z=4
        )
        
        camera_location = transform.location + offset
        camera_rotation = carla.Rotation(
            pitch=-10,
            yaw=transform.rotation.yaw
        )
        
        self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def update_camera(self):
        """Update the camera position to follow the vehicle."""
        if not self.vehicle:
            return

        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        
        # Calculate camera position
        offset = carla.Location(
            x=-15 * math.cos(yaw),
            y=15 * math.sin(yaw),
            z=4
        )
        
        camera_location = transform.location + offset
        
        # Calculate target point
        target_location = transform.location + carla.Location(
            x=5 * math.cos(yaw),
            y=5 * math.sin(yaw),
            z=0
        )
        
        # Calculate direction vector
        direction = target_location - camera_location
        
        # Calculate camera rotation
        yaw = math.degrees(math.atan2(direction.y, direction.x))
        pitch = math.degrees(math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2)))
        
        camera_rotation = carla.Rotation(
            pitch=pitch - 10,
            yaw=yaw
        )
        
        self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def create_trolley_scenario(self):
        """Create a trolley problem scenario with multiple pedestrians."""
        try:
            # Get pedestrian blueprints
            walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            
            # Create a group of pedestrians crossing the road
            spawn_point = self.vehicle.get_transform()
            road_location = spawn_point.location
            
            # Spawn pedestrians in a line across the road with spacing
            for i in range(5):
                # Position pedestrians in a line with spacing
                pedestrian_location = carla.Location(
                    x=road_location.x + i * 30.0,  # Increased spacing to 30 meters
                    y=road_location.y + 60.0,     # Increased distance ahead to 60 meters
                    z=road_location.z
                )
                
                # Check for collisions before spawning
                collision = False
                for actor in self.world.get_actors():
                    if actor.get_location().distance(pedestrian_location) < 30.0:  # Increased collision check distance
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
                        self.pedestrians.append(walker)
                        # Add AI controller for the pedestrian
                        controller = self.world.spawn_actor(
                            self.world.get_blueprint_library().find('controller.ai.walker'),
                            carla.Transform(),
                            walker
                        )
                        if controller is not None:
                            self.walker_controllers.append(controller)
                            controller.start()
                            # Make pedestrian walk across the road
                            target_location = carla.Location(
                                x=pedestrian_location.x,
                                y=pedestrian_location.y + 30.0,  # Walk 30 meters across
                                z=pedestrian_location.z
                            )
                            controller.go_to_location(target_location)
                            controller.set_max_speed(1.4)  # Walking speed
            
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
            
            # Spawn ego vehicle
            if not self.spawn_vehicle():
                raise RuntimeError("Failed to spawn ego vehicle")
            
            # Set up camera
            self.setup_camera()
            
            # Main simulation loop
            self.running = True
            while self.running:
                try:
                    # Tick the world
                    self.world.tick()
                    
                    # Get sensor data
                    sensor_data = self.sensor_manager.get_sensor_data()
                    if not sensor_data or 'camera' not in sensor_data:
                        print("Warning: No camera data available")
                        continue
                    
                    # Process sensor data
                    try:
                        features = self.ml_manager.process_sensor_data(sensor_data)
                    except Exception as e:
                        print(f"Error processing sensor data: {e}")
                        continue
                    
                    # Make decision
                    try:
                        decision = self.ml_manager.make_decision(features)
                    except Exception as e:
                        print(f"Error making decision: {e}")
                        continue
                    
                    # Apply controls
                    try:
                        self.vehicle.apply_control(carla.VehicleControl(
                            throttle=float(decision.get('throttle', 0.0)),
                            brake=float(decision.get('brake', 0.0)),
                            steer=float(decision.get('steer', 0.0))
                        ))
                    except Exception as e:
                        print(f"Error applying controls: {e}")
                        continue
                    
                    # Update camera
                    try:
                        self.update_camera()
                    except Exception as e:
                        print(f"Error updating camera: {e}")
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
                    actor.destroy()
            
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
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 