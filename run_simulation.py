import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Add CARLA Python API to the Python path
carla_path = os.path.join(project_root, "carla-python-api")
if os.path.exists(carla_path):
    sys.path.append(carla_path)
else:
    print(f"Error: CARLA path not found at {carla_path}")
    print("Please make sure CARLA is installed and update the carla_path variable")
    sys.exit(1)

# Now import and run the simulation
from src.simulation.carla_simulator import CarlaSimulator

if __name__ == "__main__":
    simulator = CarlaSimulator()
    simulator.run() 