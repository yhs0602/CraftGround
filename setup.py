from setuptools import setup, find_packages

setup(
    name="craftground",
    version="1.7",
    packages=find_packages(),
    install_requires=["gymnasium", "Pillow", "numpy", "protobuf", "typing_extensions"],
    author="yhs0602",
    author_email="jourhyang123@gmail.com",
    description="Lightweight Minecraft Environment for Reinforcement Learning",
    license="MIT",
    keywords="minecraft, reinforcement learning, environment",
    url="https://github.com/yhs0602/CraftGround",
    package_data={
        "craftground": ["craftground/MinecraftEnv/*"],
    },
)
