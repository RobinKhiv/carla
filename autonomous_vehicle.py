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

        # Choose a vehicle blueprint
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

        # Spawn the vehicle at a random spawn point
        vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))

        # Set the vehicle in autopilot mode
        vehicle.set_autopilot(True)

        print("Vehicle spawned and autopilot enabled. Press Ctrl+C to exit.")

        # Keep the script running
        while True:
            # Get the vehicle's location and transform
            location = vehicle.get_location()
            transform = vehicle.get_transform()
            
            print(f"Vehicle Location: {location}")
            print(f"Vehicle Transform: {transform}")
            
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