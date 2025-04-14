import sys
import os

# Add CARLA Python API to the Python path
carla_path = r"C:\CARLA_0.9.14\PythonAPI\carla"
if os.path.exists(carla_path):
    sys.path.append(carla_path)
else:
    print(f"Error: CARLA path not found at {carla_path}")
    print("Please make sure CARLA is installed and update the carla_path variable")
    sys.exit(1)

import carla
import random
import time

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
        # Get the vehicle's transform
        vehicle_transform = vehicle.get_transform()
        # Move the spectator behind the vehicle
        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(x=-10, z=3),
            carla.Rotation(pitch=-20, yaw=vehicle_transform.rotation.yaw)
        )
        spectator.set_transform(spectator_transform)

        print("Vehicle spawned and autopilot enabled. Press Ctrl+C to exit.")

        # Keep the script running
        while True:
            # Get the vehicle's location and transform
            location = vehicle.get_location()
            transform = vehicle.get_transform()
            
            print(f"Vehicle Location: {location}")
            print(f"Vehicle Transform: {transform}")
            
            # Update spectator position to follow the vehicle
            spectator_transform = carla.Transform(
                transform.location + carla.Location(x=-10, z=3),
                carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
            )
            spectator.set_transform(spectator_transform)
            
            time.sleep(1.0)

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