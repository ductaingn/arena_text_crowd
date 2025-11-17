from setuptools import find_packages, setup

package_name = "arena_text_crowd"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "attrs",
        "tqdm" "scipy",
        "shapely==2.1.2",
        "pyglet==1.5.0" "rasterio==1.2.10",
        "opencv-python==4.6.0.66",
        "torch",
        "transformers==4.30.2",
        "huggingface-hub==0.16.4",
        "diffusers==0.21.4",
        "google==3.0.0",
        "google-auth==2.40.3",
        "google-genai==1.32.0",
        "googleapis-common-protos==1.70.0",
        "rvo2 @ git+https://github.com/sybrenstuvel/Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d",
    ],
    zip_safe=True,
    maintainer="Duc Tai Nguyen",
    maintainer_email="ductaingn.015203@gmail.com",
    description="Arena-Text-Crowd implementation",
    license="MIT",
    tests_require=["pytest"],
)
