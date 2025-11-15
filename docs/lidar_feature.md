# Lidar Feature

## Overview

The Lidar feature provides 360-degree raycast-based distance sensing for the agent, similar to real-world LiDAR sensors used in robotics and autonomous vehicles. This is useful for:

- Navigation and obstacle avoidance
- Environment mapping
- Distance-based decision making
- Detecting entities and blocks around the agent

## Configuration

### LidarConfig Parameters

```python
from craftground import LidarConfig

lidar_config = LidarConfig(
    horizontal_rays=32,      # Number of rays in horizontal plane (360° / horizontal_rays = angle between rays)
    max_distance=100.0,      # Maximum raycast distance in blocks
    vertical_angle=0.0,      # Vertical angle offset in degrees (0.0 = horizontal)
    vertical_rays=1,         # Number of vertical layers (1 = single horizontal plane)
    vertical_fov=30.0        # Vertical field of view in degrees (used when vertical_rays > 1)
)
```

#### Parameters Explained:

- **horizontal_rays**: Number of rays distributed evenly around 360 degrees
  - Example: 32 rays = 11.25° between each ray
  - Higher values = better resolution but slower performance
  - Typical values: 16-64 for single layer, 32-128 for multi-layer

- **max_distance**: Maximum distance to raycast in blocks
  - Rays that don't hit anything return this distance
  - Typical values: 50.0-200.0 blocks
  - Lower values = better performance

- **vertical_angle**: Vertical angle offset from horizontal plane
  - 0.0 = horizontal plane at eye level
  - Positive = upward tilt
  - Negative = downward tilt
  - Only used when vertical_rays = 1

- **vertical_rays**: Number of vertical layers
  - 1 = single horizontal plane (fastest)
  - >1 = multiple layers creating a 3D scan
  - Example: 8 layers with 30° FOV = 8 layers from -15° to +15°

- **vertical_fov**: Vertical field of view when using multiple layers
  - Only used when vertical_rays > 1
  - Distributes vertical rays across this FOV range
  - Example: 30.0 = rays from -15° to +15° from center

## Usage

### Basic Usage (Single Horizontal Layer)

```python
import craftground
from craftground import LidarConfig

# Simple horizontal Lidar
lidar_config = LidarConfig(
    horizontal_rays=32,
    max_distance=100.0
)

initial_env = craftground.InitialEnvironmentConfig(
    lidar_config=lidar_config
)

env = craftground.make(initial_env_config=initial_env)
obs, info = env.reset()

# Access Lidar data
lidar_result = obs["full"].lidar_result
for ray in lidar_result.rays:
    print(f"Angle: {ray.angle_horizontal:.1f}°, Distance: {ray.distance:.2f}, Hit: {ray.hit_type}")
```

### Multi-Layer 3D Scanning

```python
# 3D Lidar configuration (like Velodyne)
lidar_config = LidarConfig(
    horizontal_rays=64,      # 64 rays per layer
    max_distance=100.0,
    vertical_rays=16,        # 16 vertical layers
    vertical_fov=30.0        # -15° to +15° vertical coverage
)
```

### Downward-Looking Lidar

```python
# Look down for ground distance measurement
lidar_config = LidarConfig(
    horizontal_rays=16,
    max_distance=50.0,
    vertical_angle=-45.0,    # Look 45° downward
    vertical_rays=1
)
```

## Observation Structure

The Lidar result is available in `obs["full"].lidar_result`:

```python
lidar_result = obs["full"].lidar_result

# Metadata
lidar_result.horizontal_rays  # int: Number of horizontal rays
lidar_result.vertical_rays    # int: Number of vertical layers
lidar_result.max_distance     # float: Maximum distance

# Ray data (list of rays)
for ray in lidar_result.rays:
    ray.distance            # float: Distance to hit (or max_distance if miss)
    ray.hit_type            # int: 0=MISS, 1=BLOCK, 2=ENTITY
    ray.block_name          # str: Block translation key (if hit_type==1)
    ray.entity_name         # str: Entity translation key (if hit_type==2)
    ray.angle_horizontal    # float: Horizontal angle in degrees (0-360)
    ray.angle_vertical      # float: Vertical angle in degrees
```

## Processing Lidar Data

### Convert to NumPy Arrays

```python
import numpy as np

lidar_result = obs["full"].lidar_result

# Extract distances
distances = np.array([ray.distance for ray in lidar_result.rays])

# Extract hit types
hit_types = np.array([ray.hit_type for ray in lidar_result.rays])

# Extract angles
angles_h = np.array([ray.angle_horizontal for ray in lidar_result.rays])
angles_v = np.array([ray.angle_vertical for ray in lidar_result.rays])

# Find closest obstacle
min_distance = np.min(distances)
min_angle = angles_h[np.argmin(distances)]
print(f"Closest obstacle at {min_distance:.2f} blocks, angle {min_angle:.1f}°")
```

### Reshape for Multi-Layer Lidar

```python
# For multi-layer Lidar, reshape into 2D array
h_rays = lidar_result.horizontal_rays
v_rays = lidar_result.vertical_rays

distances_2d = distances.reshape(v_rays, h_rays)
# Now distances_2d[v][h] gives distance at vertical layer v, horizontal angle h
```

### Convert to Point Cloud

```python
import numpy as np

def lidar_to_pointcloud(lidar_result, player_yaw, player_pitch):
    """Convert Lidar rays to 3D point cloud"""
    points = []
    
    for ray in lidar_result.rays:
        if ray.hit_type == 0:  # Skip misses
            continue
            
        # Calculate 3D position
        angle_h_rad = np.radians(player_yaw + ray.angle_horizontal)
        angle_v_rad = np.radians(player_pitch + ray.angle_vertical)
        
        x = ray.distance * np.cos(angle_v_rad) * np.sin(angle_h_rad)
        y = ray.distance * np.sin(angle_v_rad)
        z = ray.distance * np.cos(angle_v_rad) * np.cos(angle_h_rad)
        
        points.append([x, y, z])
    
    return np.array(points)

# Usage
obs, info = env.reset()
player_yaw = obs["full"].yaw
player_pitch = obs["full"].pitch
pointcloud = lidar_to_pointcloud(obs["full"].lidar_result, player_yaw, player_pitch)
```

## Performance Considerations

Lidar raycast can be computationally expensive:

- **Total rays** = `horizontal_rays × vertical_rays`
- Each ray performs a raycast through the world

### Performance Tips:

1. **Start with fewer rays**: Use 16-32 horizontal rays for testing
2. **Reduce max_distance**: Shorter raycasts are faster
3. **Use single layer first**: vertical_rays=1 is much faster than multi-layer
4. **Profile your application**: Measure actual performance impact

### Recommended Configurations:

**Fast (Real-time RL training)**:
```python
LidarConfig(horizontal_rays=16, max_distance=50.0, vertical_rays=1)
```

**Balanced**:
```python
LidarConfig(horizontal_rays=32, max_distance=100.0, vertical_rays=1)
```

**High Resolution**:
```python
LidarConfig(horizontal_rays=64, max_distance=100.0, vertical_rays=1)
```

**3D Scanning**:
```python
LidarConfig(horizontal_rays=64, max_distance=100.0, vertical_rays=16, vertical_fov=30.0)
```

## Example Use Cases

### 1. Obstacle Avoidance

```python
def get_safe_direction(lidar_result):
    """Find the direction with the most open space"""
    distances = np.array([ray.distance for ray in lidar_result.rays])
    angles = np.array([ray.angle_horizontal for ray in lidar_result.rays])
    
    # Find direction with maximum distance
    best_idx = np.argmax(distances)
    return angles[best_idx]
```

### 2. Wall Following

```python
def detect_walls(lidar_result, threshold=5.0):
    """Detect nearby walls"""
    walls = []
    for ray in lidar_result.rays:
        if ray.hit_type == 1 and ray.distance < threshold:
            walls.append((ray.angle_horizontal, ray.distance))
    return walls
```

### 3. Entity Detection

```python
def find_nearest_entity(lidar_result):
    """Find nearest entity"""
    entities = [(ray.distance, ray.angle_horizontal, ray.entity_name) 
                for ray in lidar_result.rays if ray.hit_type == 2]
    if entities:
        return min(entities, key=lambda x: x[0])
    return None
```

## Technical Details

- Raycasts are performed from the player's eye position
- Horizontal angles are relative to player's yaw (0° = forward)
- Vertical angles are relative to player's pitch (0° = level)
- Block and entity hits are both detected
- Closest hit (block or entity) is returned for each ray
- Ray order: Iterate through vertical layers, then horizontal angles within each layer

## See Also

- [Observation Space Documentation](observation_space/index.md)
- [Initial Environment Configuration](configuration/initial_environment.md)
- Example: `examples/lidar_example.py`

