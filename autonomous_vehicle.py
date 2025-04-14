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

        # Add a debug camera to the vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Attach camera to the vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("Debug camera attached to vehicle")

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
            # Keep the camera at a fixed angle relative to the vehicle's movement
            offset = carla.Location(
                x=-15 * math.cos(yaw),  # 15 meters behind the vehicle
                y=15 * math.sin(yaw),   # 15 meters behind the vehicle
                z=4                     # 4 meters above the vehicle
            )
            
            # Set camera position
            camera_location = transform.location + offset
            
            # Calculate the camera's target point (slightly ahead of the vehicle)
            target_location = transform.location + carla.Location(
                x=5 * math.cos(yaw),  # 5 meters ahead of the vehicle
                y=5 * math.sin(yaw),  # 5 meters ahead of the vehicle
                z=0                   # Same height as vehicle
            )
            
            # Calculate the direction vector from camera to target
            direction = target_location - camera_location
            
            # Calculate the yaw and pitch angles for the camera
            yaw = math.degrees(math.atan2(direction.y, direction.x))
            pitch = math.degrees(math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2)))
            
            # Set camera rotation
            camera_rotation = carla.Rotation(
                pitch=pitch - 10,  # Look slightly down at the vehicle
                yaw=yaw           # Look at the vehicle
            )
            
            # Update spectator position
            spectator.set_transform(carla.Transform(camera_location, camera_rotation))
            
            # Print vehicle info every 2 seconds
            if int(time.time()) % 2 == 0:
                print(f"Vehicle Location: {location}")
                print(f"Vehicle Speed: {speed*3.6:.2f} km/h")
                print(f"Vehicle Rotation: {transform.rotation}")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nDestroying actors...")
        if 'camera' in locals():
            camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        print("Done.")
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'camera' in locals():
            camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()

if __name__ == '__main__':
    main() 