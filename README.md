# CARLA Autonomous Vehicle Project (Windows)

This project demonstrates basic autonomous vehicle functionality using the CARLA simulator on Windows.

## Prerequisites

1. Install CARLA Simulator:
   - Download CARLA 0.9.14 from the [official website](https://github.com/carla-simulator/carla/releases/tag/0.9.14)
   - Extract the downloaded file to a location of your choice
   - Run CarlaUE4.exe from the extracted folder

2. Install Python Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the CARLA server by running CarlaUE4.exe
2. Run the autonomous vehicle script:
   ```bash
   python autonomous_vehicle.py
   ```

## Features

- Connects to CARLA server
- Spawns a random vehicle
- Enables autopilot mode
- Prints vehicle location and transform data
- Graceful error handling and cleanup

## Notes

- Make sure the CARLA server is running before starting the script
- The script connects to localhost:2000 by default
- Press Ctrl+C to stop the script and clean up the vehicle
- The script includes error handling to ensure proper cleanup of resources

## Next Steps

- Add sensor configuration (cameras, LIDAR, etc.)
- Implement custom autonomous driving logic
- Add traffic and pedestrian simulation
- Implement collision detection and avoidance 