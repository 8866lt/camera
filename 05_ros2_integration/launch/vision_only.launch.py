"""
仅视觉系统(相机+检测)
用于调试视觉部分
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    camera_share = get_package_share_directory('camera_publisher')
    detector_share = get_package_share_directory('object_detector')
    
    camera_config = os.path.join(camera_share, 'config', 'camera_params.yaml')
    detector_config = os.path.join(detector_share, 'config', 'detector_params.yaml')
    
    camera_node = Node(
        package='camera_publisher',
        executable='camera_node',
        name='camera_publisher',
        output='screen',
        parameters=[camera_config]
    )
    
    detector_node = Node(
        package='object_detector',
        executable='detector_node',
        name='object_detector',
        output='screen',
        parameters=[detector_config]
    )
    
    # 图像查看工具
    image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_viewer',
        arguments=['/detections/visualization']
    )
    
    return LaunchDescription([
        camera_node,
        detector_node,
        image_view
    ])
