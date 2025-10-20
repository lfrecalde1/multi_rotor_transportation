from setuptools import find_packages, setup

package_name = 'multi_rotor_transportation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='lfrecalde1@espe.edu.ec',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'payload_dynamics = multi_rotor_transportation.main:main',
            'payload_control = multi_rotor_transportation.main_acceleration:main',
            'payload_control_mujoco = multi_rotor_transportation.main_mujoco:main',
            'payload_simple_control = multi_rotor_transportation.main_simple_payload:main',
            'payload_simple_mujoco = multi_rotor_transportation.main_simple_mujoco:main',
        ],
    },
)
