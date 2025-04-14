# Autonomous Vehicle Simulation with Ethical Decision Making

This project implements an autonomous vehicle simulation using the CARLA simulator, focusing on ethical decision-making in complex driving scenarios. The system incorporates various sensors, AI decision-making, and an ethical engine to ensure safe and morally sound vehicle behavior.

## Project Structure

```
.
├── src/
│   ├── simulation/
│   │   └── carla_simulator.py    # Main simulation class
│   ├── sensors/
│   │   └── sensor_manager.py     # Manages vehicle sensors
│   ├── ai/
│   │   └── decision_maker.py     # AI decision-making system
│   ├── ethics/
│   │   └── ethical_engine.py     # Ethical decision-making system
│   └── utils/
│       └── sensor_utils.py       # Sensor data processing utilities
├── requirements.txt              # Project dependencies
└── README.md                    # This file
```

## Features

- **Multi-sensor Integration**: Combines data from RGB cameras, LiDAR, radar, and other sensors
- **Real-time Decision Making**: Processes sensor data to make driving decisions
- **Ethical Framework**: Implements ethical decision-making for complex scenarios
- **Risk Assessment**: Evaluates and responds to potential hazards
- **Smooth Vehicle Control**: Ensures comfortable and safe vehicle movement

## Requirements

- Python 3.7+
- CARLA Simulator 0.9.13
- Dependencies listed in `requirements.txt`

## Installation

1. Install CARLA 0.9.13:
   - Download from [CARLA's official website](https://carla.org/)
   - Extract to a directory (e.g., `C:\carla`)

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add CARLA Python API to your Python path:
   - Set the `carla_path` variable in `src/simulation/carla_simulator.py` to point to your CARLA installation's Python API

## Usage

1. Start the CARLA server:
   ```bash
   # Windows
   C:\carla\CarlaUE4.exe

   # Linux
   ./CarlaUE4.sh
   ```

2. Run the simulation:
   ```bash
   python -m src.simulation.carla_simulator
   ```

## Ethical Decision Making

The system implements an ethical framework that prioritizes:
1. Pedestrian safety (highest priority)
2. Passenger safety
3. Property damage (lowest priority)

The ethical engine evaluates decisions based on:
- Distance to pedestrians
- Risk assessment
- Potential harm to different parties
- Traffic rules and regulations

## Development

The project is organized into several key components:

1. **Simulation**: Handles the CARLA environment and vehicle control
2. **Sensors**: Manages data collection from various vehicle sensors
3. **AI**: Implements decision-making algorithms
4. **Ethics**: Handles ethical considerations in decision-making
5. **Utils**: Provides utility functions for data processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CARLA Simulator Team
- Contributors to the open-source community
- Research papers and publications in autonomous vehicle ethics 