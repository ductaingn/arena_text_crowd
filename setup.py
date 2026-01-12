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
        "tqdm",
        "scipy",
        "Shapely",
        "pyglet",
        "rasterio",
        "opencv-python",
        "tensorboard",
        "tensorboard-data-server",
        "tensorboard-plugin-wit",
        "tokenizers",
        "torch",
        "torchaudio",
        "torchmetrics",
        "torchvision",
        "transformers",
        "huggingface-hub",
        "diffusers",
        "google",
        "google-auth",
        "google-genai",
        "googleapis-common-protos",
    ],
    zip_safe=True,
    maintainer="Duc Tai Nguyen",
    maintainer_email="ductaingn.015203@gmail.com",
    description="Arena-Text-Crowd implementation",
    license="MIT",
)
