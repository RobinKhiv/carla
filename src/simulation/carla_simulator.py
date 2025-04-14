import carla
import time
import math
import numpy as np
import random
from typing import List, Dict, Any
from ..sensors.sensor_manager import SensorManager
from ..ai.decision_maker import DecisionMaker
from ..ethics.ethical_engine import EthicalEngine

class CarlaSimulator:
    def __init__(self, host: str = 'localhost', port: int = 2000):
        """Initialize the CARLA simulator with connection parameters."""
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = None
        self.vehicle = None
        self.sensor_manager = None
        self.decision_maker = None
        self.ethical_engine = None
        self.spectator = None
        self.running = False
        self.pedestrians = []
        self.other_vehicles = []
        self.walker_controllers = []
        self.traffic_manager = None

    def initialize(self):
        """Initialize the simulation environment and components."""
        try:
            # Get the world
            self.world = self.client.get_world()
            
            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            
            # Initialize components
            self.sensor_manager = SensorManager(self.world)
            self.decision_maker = DecisionMaker()
            self.ethical_engine = EthicalEngine()
            
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
            for i in range(num_pedestrians):
                spawn_point = carla.Transform()
                spawn_point.location = self.world.get_random_location_from_navigation()
                if spawn_point.location is not None:
                    spawn_points.append(spawn_point)
            
            # Spawn pedestrians
            for spawn_point in spawn_points:
                walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                self.pedestrians.append(walker)
            
            # Spawn walker controllers
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            for walker in self.pedestrians:
                controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                self.walker_controllers.append(controller)
                controller.start()
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
            
            # Spawn pedestrians in a line across the road
            for i in range(5):
                # Position pedestrians in a line
                pedestrian_location = carla.Location(
                    x=road_location.x + i * 2.0,  # 2 meters apart
                    y=road_location.y + 5.0,      # 5 meters ahead
                    z=road_location.z
                )
                
                # Create transform for pedestrian
                pedestrian_transform = carla.Transform(
                    pedestrian_location,
                    carla.Rotation(yaw=90.0)  # Facing across the road
                )
                
                # Spawn pedestrian
                walker = self.world.spawn_actor(random.choice(walker_bp), pedestrian_transform)
                self.pedestrians.append(walker)
                
                # Add AI controller
                controller = self.world.spawn_actor(
                    self.world.get_blueprint_library().find('controller.ai.walker'),
                    carla.Transform(),
                    walker
                )
                self.walker_controllers.append(controller)
                controller.start()
                
                # Make pedestrians walk across the road
                target_location = carla.Location(
                    x=pedestrian_location.x,
                    y=pedestrian_location.y + 10.0,  # Walk 10 meters across
                    z=pedestrian_location.z
                )
                controller.go_to_location(target_location)
                controller.set_max_speed(1.4)  # Walking speed
            
            print("Created trolley problem scenario with 5 pedestrians crossing the road")
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
                x=spawn_point.location.x + 30.0,  # 30 meters ahead
                y=spawn_point.location.y,
                z=spawn_point.location.z
            )
            
            # Spawn broken-down vehicle
            hazard_vehicle = self.world.spawn_actor(
                random.choice(vehicle_bps),
                carla.Transform(hazard_location, spawn_point.rotation)
            )
            self.other_vehicles.append(hazard_vehicle)
            
            # Create debris around the vehicle
            for i in range(3):
                debris_location = carla.Location(
                    x=hazard_location.x + random.uniform(-2.0, 2.0),
                    y=hazard_location.y + random.uniform(-2.0, 2.0),
                    z=hazard_location.z
                )
                
                # Spawn static obstacle
                obstacle = self.world.spawn_actor(
                    self.world.get_blueprint_library().find('static.prop.container'),
                    carla.Transform(debris_location)
                )
                self.other_vehicles.append(obstacle)
            
            print("Created hazard scenario with broken-down vehicle and debris")
        except Exception as e:
            print(f"Error creating hazard scenario: {e}")

    def run(self):
        """Run the main simulation loop."""
        if not self.initialize():
            return

        if not self.spawn_vehicle():
            return

        # Spawn regular traffic
        self.spawn_pedestrians(10)
        self.spawn_other_vehicles(10)

        # Create special scenarios
        self.create_trolley_scenario()
        self.create_hazard_scenario()

        self.setup_camera()
        self.running = True

        try:
            while self.running:
                # Get sensor data
                sensor_data = self.sensor_manager.get_sensor_data()
                
                # Make decisions based on sensor data and ethical considerations
                decision = self.decision_maker.make_decision(sensor_data)
                
                # Apply ethical constraints
                ethical_decision = self.ethical_engine.evaluate_decision(decision, sensor_data)
                
                # Apply the decision to the vehicle
                self.apply_decision(ethical_decision)
                
                # Update camera
                self.update_camera()
                
                # Sleep to maintain real-time simulation
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        finally:
            self.cleanup()

    def apply_decision(self, decision: Dict[str, Any]):
        """Apply the decision to the vehicle."""
        if not self.vehicle:
            return

        # Apply control commands to the vehicle
        control = carla.VehicleControl()
        control.throttle = decision.get('throttle', 0.0)
        control.steer = decision.get('steer', 0.0)
        control.brake = decision.get('brake', 0.0)
        
        self.vehicle.apply_control(control)

    def cleanup(self):
        """Clean up resources and destroy actors."""
        print("\nCleaning up...")
        
        # Stop walker controllers
        for controller in self.walker_controllers:
            controller.stop()
        
        # Destroy all actors
        for actor in self.pedestrians + self.other_vehicles:
            if actor is not None:
                actor.destroy()
        
        if self.sensor_manager:
            self.sensor_manager.destroy()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        print("Cleanup complete")

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 