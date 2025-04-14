import numpy as np
import carla
from typing import Dict, Any, List

def process_image(image: carla.Image) -> np.ndarray:
    """Process raw camera image data."""
    # Convert to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha channel
    return array

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

def process_radar(radar_data: carla.RadarMeasurement) -> Dict[str, Any]:
    """Process radar measurement data."""
    points = []
    velocities = []
    
    for detection in radar_data:
        points.append([detection.depth * np.cos(detection.azimuth),
                      detection.depth * np.sin(detection.azimuth),
                      detection.altitude])
        velocities.append(detection.velocity)
    
    points = np.array(points)
    velocities = np.array(velocities)
    
    return {
        'points': points,
        'velocities': velocities,
        'mean_velocity': np.mean(velocities) if len(velocities) > 0 else 0.0
    }

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
            if not current_cluster:
                current_cluster = [point]
            else:
                # Check if point is close to current cluster
                last_point = current_cluster[-1]
                if np.linalg.norm(point - last_point) < 1.0:
                    current_cluster.append(point)
                else:
                    if len(current_cluster) > 5:  # Minimum cluster size
                        clusters.append(current_cluster)
                    current_cluster = [point]
    
    # Process clusters
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

def calculate_risk_score(sensor_data: Dict[str, Any]) -> float:
    """Calculate a risk score based on sensor data."""
    risk_score = 0.0
    
    # Check for collisions
    if 'collision' in sensor_data:
        risk_score += 1.0
    
    # Check for lane invasions
    if 'lane_invasion' in sensor_data:
        risk_score += 0.5
    
    # Check for nearby obstacles
    if 'lidar' in sensor_data:
        obstacles = detect_obstacles(sensor_data['lidar'])
        for obstacle in obstacles:
            if obstacle['size'] > 1.0:  # Significant obstacle
                risk_score += 0.3
    
    # Check for high relative velocities
    if 'radar' in sensor_data:
        mean_velocity = sensor_data['radar']['mean_velocity']
        if abs(mean_velocity) > 10.0:  # High relative velocity
            risk_score += 0.2
    
    return min(risk_score, 1.0)  # Cap at 1.0 