import sys
import os

# Add CARLA Python API to the Python path
carla_path = r"C:\carla\PythonAPI\carla"
if os.path.exists(carla_path):
    sys.path.append(carla_path)
else:
    print(f"Error: CARLA path not found at {carla_path}")
    print("Please make sure CARLA is installed and update the carla_path variable")
    sys.exit(1)

import carla
import random
import time
import math

def main():
    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get the world
        world = client.get_world()

        # Get the blueprint library
        blueprint_library = world.get_blueprint_library()

        # Get the map's spawn points
        spawn_points = world.get_map().get_spawn_points()
        
        # Choose a specific spawn point (the first one is usually in a good location)
        spawn_point = spawn_points[0]
        print(f"Spawning vehicle at: {spawn_point.location}")

        # Choose a vehicle blueprint (let's use a specific vehicle for visibility)
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if vehicle_bp is None:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

        # Spawn the vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned: {vehicle}")

        # Set the vehicle in autopilot mode
        vehicle.set_autopilot(True)
        print("Autopilot enabled")

        # Set up spectator camera
        spectator = world.get_spectator()
        
        print("Vehicle spawned and autopilot enabled. Press Ctrl+C to exit.")

        # Keep the script running
        while True:
            # Get the vehicle's location and transform
            location = vehicle.get_location()
            transform = vehicle.get_transform()
            
            # Calculate vehicle speed
            vehicle_velocity = vehicle.get_velocity()
            speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
            
            # Calculate the camera position based on vehicle's current position and rotation
            yaw = math.radians(transform.rotation.yaw)
            # Calculate the offset vector based on vehicle's rotation
            # Note: In CARLA, yaw=0 points to the right, so we need to adjust our calculations
            offset = carla.Location(
                x=15 * math.cos(yaw),   # 15 meters behind the vehicle
                y=15 * math.sin(yaw),   # 15 meters behind the vehicle
                z=8                     # 8 meters above the vehicle
            )
            
            # Set camera position and rotation
            camera_location = transform.location + offset
            camera_rotation = carla.Rotation(
                pitch=-15,  # Look down at 15 degrees
                yaw=transform.rotation.yaw  # Match vehicle's yaw
            )
            
            # Update spectator position
            spectator.set_transform(carla.Transform(camera_location, camera_rotation))
            
            # Print vehicle info every 2 seconds
            if int(time.time()) % 2 == 0:
                print(f"Vehicle Location: {location}")
                print(f"Vehicle Speed: {speed*3.6:.2f} km/h")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nDestroying actors...")
        if 'vehicle' in locals():
            vehicle.destroy()
        print("Done.")
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'vehicle' in locals():
            vehicle.destroy()

if __name__ == '__main__':
    main() 