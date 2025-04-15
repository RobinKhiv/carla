import numpy as np
import carla
from typing import Dict, Any, List

class SensorUtils:
    """Utility class for processing sensor data."""
    
    @staticmethod
    def process_image(image: carla.Image) -> np.ndarray:
        """Process raw camera image data."""
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        return array

    @staticmethod
    def process_lidar(point_cloud: carla.LidarMeasurement) -> Dict[str, Any]:
        """Process LiDAR point cloud data."""
        points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        
        # Calculate basic statistics
        distances = np.sqrt(np.sum(points**2, axis=1))
        
        return {
            'points': points,
            'distances': distances,
            'mean_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }

    @staticmethod
    def process_radar(radar_data: carla.RadarMeasurement) -> Dict[str, Any]:
        """Process radar measurement data."""
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Extract velocity information
        velocities = np.sqrt(np.sum(points[:, 2:]**2, axis=1))
        
        return {
            'points': points,
            'velocities': velocities,
            'mean_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities)
        }

    @staticmethod
    def detect_obstacles(lidar_data: Dict[str, Any], threshold: float = 5.0) -> List[Dict[str, Any]]:
        """Detect obstacles from LiDAR data."""
        obstacles = []
        points = lidar_data['points']
        distances = lidar_data['distances']
        
        # Group points into clusters
        clusters = []
        current_cluster = []
        
        for i, (point, distance) in enumerate(zip(points, distances)):
            if distance < threshold:
                if not current_cluster or np.linalg.norm(point - current_cluster[-1]) < 1.0:
                    current_cluster.append(point)
                else:
                    if len(current_cluster) > 10:  # Minimum points for a valid obstacle
                        clusters.append(current_cluster)
                    current_cluster = [point]
        
        if current_cluster and len(current_cluster) > 10:
            clusters.append(current_cluster)
        
        # Process each cluster
        for cluster in clusters:
            cluster_points = np.array(cluster)
            center = np.mean(cluster_points, axis=0)
            size = np.max(np.linalg.norm(cluster_points - center, axis=1))
            
            obstacles.append({
                'center': center,
                'size': size,
                'points': cluster_points
            })
        
        return obstacles

    @staticmethod
    def calculate_risk_score(sensor_data: Dict[str, Any]) -> float:
        """Calculate a risk score based on sensor data."""
        risk_score = 0.0
        
        # Check LiDAR data
        if 'lidar' in sensor_data:
            lidar_data = sensor_data['lidar']
            if lidar_data['min_distance'] < 5.0:
                risk_score += 0.5
            if lidar_data['mean_distance'] < 10.0:
                risk_score += 0.3
        
        # Check radar data
        if 'radar' in sensor_data:
            radar_data = sensor_data['radar']
            if radar_data['max_velocity'] > 5.0:
                risk_score += 0.2
        
        # Check camera data
        if 'camera' in sensor_data:
            # Add risk based on detected objects
            # This is a placeholder - actual implementation would use object detection
            risk_score += 0.1
        
        return min(risk_score, 1.0)  # Cap risk score at 1.0

    @staticmethod
    def extract_features(sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from sensor data for ML models."""
        features = []
        
        # Extract LiDAR features
        if 'lidar' in sensor_data:
            lidar_data = sensor_data['lidar']
            features.extend([
                lidar_data['mean_distance'],
                lidar_data['min_distance'],
                lidar_data['max_distance']
            ])
        
        # Extract radar features
        if 'radar' in sensor_data:
            radar_data = sensor_data['radar']
            features.extend([
                radar_data['mean_velocity'],
                radar_data['max_velocity']
            ])
        
        # Extract camera features (placeholder)
        if 'camera' in sensor_data:
            features.extend([0.0, 0.0, 0.0])  # Placeholder for actual features
        
        return np.array(features) 