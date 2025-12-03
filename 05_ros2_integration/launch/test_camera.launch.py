"""
测试相机Launch文件
快速测试相机是否正常工作
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    device_id_arg = DeclareLaunchArgument(
        'device_id',
        default_value='0',
        description='相机设备ID'
    )
    
    camera_node = Node(
        package='camera_publisher',
        executable='camera_node',
        name='camera_test',
        output='screen',
        parameters=[{
            'device_id': LaunchConfiguration('device_id'),
            'frame_rate': 30,
            'width': 640,
            'height': 480,
            'publish_compressed': False
        }]
    )
    
    image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_viewer'
    )
    
    return LaunchDescription([
        device_id_arg,
        camera_node,
        image_view
    ])
