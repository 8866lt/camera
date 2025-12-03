from setuptools import setup
import os
from glob import glob

package_name = 'visual_servo'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='视觉伺服控制节点包',
    license='MIT',
    entry_points={
        'console_scripts': [
            'servo_node = visual_servo.servo_node:main',
            'depth_estimator = visual_servo.depth_estimator:main',
        ],
    },
)
