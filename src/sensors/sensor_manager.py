import carla
import numpy as np
from typing import Dict, Any, List
from ..utils.sensor_utils import SensorUtils

class SensorManager:
    def __init__(self, world: carla.World):
        """Initialize the sensor manager."""
        self.world = world
        self.sensors = {}
        self.sensor_data = {}
        self.sensor_utils = SensorUtils()
        
    def attach_sensors(self, vehicle: carla.Vehicle):
        """Attach sensors to the vehicle."""
        # Attach RGB camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self.sensors['camera'] = camera
        camera.listen(lambda image: self._on_camera_data(image, 'camera'))
        
        # Attach LiDAR
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        self.sensors['lidar'] = lidar
        lidar.listen(lambda point_cloud: self._on_lidar_data(point_cloud, 'lidar'))
        
        # Attach radar
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('points_per_second', '1500')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
        self.sensors['radar'] = radar
        radar.listen(lambda radar_data: self._on_radar_data(radar_data, 'radar'))
        
        # Collision sensor
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        self.sensors['collision'] = collision
        collision.listen(lambda event: self._on_collision(event))

        # Lane invasion sensor
        lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_invasion = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=vehicle)
        self.sensors['lane_invasion'] = lane_invasion
        lane_invasion.listen(lambda event: self._on_lane_invasion(event))

    def _on_camera_data(self, image: carla.Image, sensor_name: str):
        """Process camera data."""
        self.sensor_data[sensor_name] = self.sensor_utils.process_image(image)

    def _on_lidar_data(self, point_cloud: carla.LidarMeasurement, sensor_name: str):
        """Process LiDAR data."""
        self.sensor_data[sensor_name] = self.sensor_utils.process_lidar(point_cloud)

    def _on_radar_data(self, radar_data: carla.RadarMeasurement, sensor_name: str):
        """Process radar data."""
        self.sensor_data[sensor_name] = self.sensor_utils.process_radar(radar_data)

    def _on_collision(self, event: carla.CollisionEvent):
        """Handle collision events."""
        self.sensor_data['collision'] = {
            'actor': event.other_actor,
            'impulse': event.normal_impulse
        }

    def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
        """Handle lane invasion events."""
        self.sensor_data['lane_invasion'] = {
            'crossed_lane_markings': event.crossed_lane_markings
        }

    def get_sensor_data(self) -> Dict[str, Any]:
        """Get the latest sensor data."""
        return self.sensor_data

    def destroy(self):
        """Destroy all sensors."""
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.destroy()
        self.sensors.clear()
        self.sensor_data.clear() 