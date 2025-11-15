"""
Example showing how to use Lidar functionality in CraftGround.

This example demonstrates:
1. Creating a LidarConfig with custom settings
2. Passing it to InitialEnvironmentConfig
3. Accessing lidar data from observations
"""

import craftground
from craftground import LidarConfig

# Create Lidar configuration
# - horizontal_rays: Number of rays around 360 degrees (higher = more resolution, slower)
# - max_distance: Maximum raycast distance in blocks
# - vertical_angle: Vertical angle offset (0.0 = horizontal plane)
# - vertical_rays: Number of vertical layers (1 = single horizontal plane)
# - vertical_fov: Vertical field of view when using multiple vertical rays
lidar_config = LidarConfig(
    horizontal_rays=32,  # 32 rays around 360 degrees (11.25° between each ray)
    max_distance=100.0,  # Raycast up to 100 blocks
    vertical_angle=0.0,  # Horizontal plane
    vertical_rays=1,  # Single horizontal layer
    vertical_fov=30.0,  # Not used when vertical_rays=1
)

# For a multi-layer Lidar (like a real Velodyne), use:
# lidar_config_multilayer = LidarConfig(
#     horizontal_rays=64,
#     max_distance=100.0,
#     vertical_angle=0.0,
#     vertical_rays=8,       # 8 vertical layers
#     vertical_fov=30.0      # Covers -15 to +15 degrees vertically
# )

# Create initial environment config with Lidar
initial_env = craftground.InitialEnvironmentConfig(
    image_width=640,
    image_height=360,
    lidar_config=lidar_config,  # Enable Lidar
)

# Create environment
env = craftground.make(initial_env_config=initial_env)

# Reset environment
obs, info = env.reset()

# Access Lidar data from observation
lidar_result = obs["full"].lidar_result

print(f"Lidar Configuration:")
print(f"  Horizontal rays: {lidar_result.horizontal_rays}")
print(f"  Vertical rays: {lidar_result.vertical_rays}")
print(f"  Max distance: {lidar_result.max_distance}")
print(f"  Total rays: {len(lidar_result.rays)}")

# Iterate through all rays
for i, ray in enumerate(lidar_result.rays):
    if ray.hit_type == 0:  # MISS
        print(
            f"Ray {i}: MISS at angle {ray.angle_horizontal:.1f}° (distance: {ray.distance:.2f})"
        )
    elif ray.hit_type == 1:  # BLOCK
        print(
            f"Ray {i}: HIT BLOCK '{ray.block_name}' at angle {ray.angle_horizontal:.1f}° (distance: {ray.distance:.2f})"
        )
    elif ray.hit_type == 2:  # ENTITY
        print(
            f"Ray {i}: HIT ENTITY '{ray.entity_name}' at angle {ray.angle_horizontal:.1f}° (distance: {ray.distance:.2f})"
        )

# Convert Lidar data to numpy array for processing
import numpy as np

distances = np.array([ray.distance for ray in lidar_result.rays])
hit_types = np.array([ray.hit_type for ray in lidar_result.rays])
angles = np.array([ray.angle_horizontal for ray in lidar_result.rays])

print(f"\nLidar Statistics:")
print(f"  Mean distance: {np.mean(distances):.2f} blocks")
print(f"  Min distance: {np.min(distances):.2f} blocks")
print(f"  Max distance: {np.max(distances):.2f} blocks")
print(f"  Hits: {np.sum(hit_types > 0)} / {len(hit_types)}")

# Run environment loop
for step in range(100):
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Process Lidar data each step
    lidar_result = obs["full"].lidar_result
    distances = np.array([ray.distance for ray in lidar_result.rays])

    if step % 10 == 0:
        print(f"Step {step}: Mean distance = {np.mean(distances):.2f} blocks")

    if terminated or truncated:
        obs, info = env.reset()

env.close()
