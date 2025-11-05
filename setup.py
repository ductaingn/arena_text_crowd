from setuptools import find_packages, setup

package_name = 'arena_hunav_sim_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Duc Tai Nguyen',
    maintainer_email='ductaingn.015203@gmail.com',
    description='Arena-HuNavSim bridge',
    license='MIT',
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': [
    #         'hello_world_node = arena_hunav_sim_bridge.hello_world_node:main'
    #     ],
    # },
)
