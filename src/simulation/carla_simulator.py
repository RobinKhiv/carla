import carla
import time
import math
import numpy as np
import random
import cv2
from typing import List, Dict, Any

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
            self.pedestrians = []
            self.other_vehicles = []
            self.walker_controllers = []
            
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
            
            # Initialize traffic manager
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
            
            print("Simulator initialized successfully with Town01 map")
            return True
        except Exception as e:
            print(f"Error initializing simulator: {e}")
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
                            
                            # Set vehicle to manual control
                            self.vehicle.set_autopilot(False)
                            
                            # Register vehicle with traffic manager
                            self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)  # Always obey traffic lights
                            self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 0)  # Maintain normal speed
                            self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 2.0)  # Safe following distance
                            self.traffic_manager.auto_lane_change(self.vehicle, False)  # Disable automatic lane changes
                            
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

    def spawn_traffic(self, num_vehicles: int = 10, num_pedestrians: int = 20):
        """Spawn traffic using CARLA's built-in traffic manager."""
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
            
            # Spawn pedestrians using CARLA's built-in pedestrian manager
            try:
                # Get pedestrian blueprints
                walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
                
                # Get spawn points for pedestrians
                spawn_points = []
                for _ in range(num_pedestrians):
                    spawn_point = carla.Transform()
                    spawn_point.location = self.world.get_random_location_from_navigation()
                    if spawn_point.location is not None:
                        spawn_points.append(spawn_point)
                
                # Spawn pedestrians
                for spawn_point in spawn_points:
                    try:
                        walker = self.world.spawn_actor(random.choice(walker_bp), spawn_point)
                        if walker is not None:
                            # Set pedestrian to be non-collidable with other pedestrians
                            walker.set_simulate_physics(False)
                            # Set random destination
                            destination = self.world.get_random_location_from_navigation()
                            if destination is not None:
                                walker.set_location(destination)
                    except Exception as e:
                        print(f"Warning: Failed to spawn pedestrian: {e}")
                        continue
                
                print(f"Spawned {num_vehicles} vehicles and {num_pedestrians} pedestrians")
            except Exception as e:
                print(f"Error spawning pedestrians: {e}")
            
        except Exception as e:
            print(f"Error spawning traffic: {e}")

    def run(self):
        """Run the simulation."""
        try:
            if not self.initialized:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize simulator")
                self.initialized = True
            
            # Clean up any existing actors
            self.cleanup()
            
            # Spawn ego vehicle
            print("Spawning ego vehicle...")
            if not self.spawn_vehicle():
                raise RuntimeError("Failed to spawn ego vehicle")
            
            # Spawn traffic
            print("Spawning traffic...")
            self.spawn_traffic(10, 20)  # Spawn 10 vehicles and 20 pedestrians
            
            # Set up camera
            if not self.setup_camera():
                print("Warning: Camera setup failed, continuing without camera")
            
            # Main simulation loop
            self.running = True
            
            while self.running:
                try:
                    # Tick the world
                    self.world.tick()
                    
                    # Update camera position
                    self.update_camera()
                    
                    # Get current waypoint
                    current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
                    if current_waypoint is None:
                        print("Warning: Vehicle is not on road")
                        continue
                    
                    # Get next waypoint
                    next_waypoint = current_waypoint.next(5.0)[0]  # Look 5 meters ahead
                    
                    # Calculate basic controls
                    control = carla.VehicleControl()
                    
                    # Get current velocity
                    current_velocity = self.vehicle.get_velocity().length() * 3.6  # Convert to km/h
                    
                    # Check for traffic lights
                    vehicle_location = self.vehicle.get_location()
                    vehicle_transform = self.vehicle.get_transform()
                    
                    # Get all traffic lights in the world
                    traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
                    
                    # Find the nearest traffic light in front of the vehicle
                    nearest_light = None
                    min_distance = float('inf')
                    
                    for light in traffic_lights:
                        # Get traffic light location
                        light_location = light.get_location()
                        
                        # Calculate distance to traffic light
                        distance = vehicle_location.distance(light_location)
                        
                        # Check if traffic light is in front of the vehicle
                        vehicle_forward = vehicle_transform.get_forward_vector()
                        light_direction = light_location - vehicle_location
                        light_direction = light_direction.make_unit_vector()
                        
                        if vehicle_forward.dot(light_direction) > 0.5 and distance < 30.0:  # Only consider lights within 30 meters in front
                            if distance < min_distance:
                                min_distance = distance
                                nearest_light = light
                    
                    # If we found a traffic light, check its state
                    if nearest_light and min_distance < 10.0:  # Only consider lights within 10 meters
                        state = nearest_light.get_state()
                        if state == carla.TrafficLightState.Red or state == carla.TrafficLightState.Yellow:
                            print(f"Traffic light is {state}, stopping...")
                            control.throttle = 0.0
                            control.brake = 1.0
                            control.steer = 0.0
                            self.vehicle.apply_control(control)
                            continue
                    
                    # Set target speed
                    target_speed = 20.0  # km/h
                    
                    # Calculate throttle and brake
                    if current_velocity < target_speed:
                        control.throttle = 0.5
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.brake = 0.1
                    
                    # Calculate steering
                    vehicle_transform = self.vehicle.get_transform()
                    vehicle_location = vehicle_transform.location
                    waypoint_location = next_waypoint.transform.location
                    
                    # Calculate vector to waypoint
                    waypoint_vector = carla.Location(
                        waypoint_location.x - vehicle_location.x,
                        waypoint_location.y - vehicle_location.y,
                        0
                    )
                    
                    # Calculate angle between vehicle's forward vector and waypoint vector
                    vehicle_forward = vehicle_transform.get_forward_vector()
                    angle = math.degrees(math.acos(
                        (vehicle_forward.x * waypoint_vector.x + vehicle_forward.y * waypoint_vector.y) /
                        (math.sqrt(vehicle_forward.x**2 + vehicle_forward.y**2) * 
                         math.sqrt(waypoint_vector.x**2 + waypoint_vector.y**2))
                    ))
                    
                    # Calculate cross product to determine direction
                    cross = vehicle_forward.x * waypoint_vector.y - vehicle_forward.y * waypoint_vector.x
                    if cross < 0:
                        angle = -angle
                    
                    # Set steering based on angle
                    control.steer = angle / 70.0  # 70 degrees is approximately the maximum steering angle
                    
                    # Apply control to vehicle
                    self.vehicle.apply_control(control)
                    
                    # Print vehicle state
                    print(f"Vehicle velocity: {current_velocity:.2f} km/h")
                    print(f"Vehicle position: {vehicle_location}")
                    print(f"Vehicle rotation: {vehicle_transform.rotation}")
                    
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