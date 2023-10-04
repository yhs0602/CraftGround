from setuptools import setup, find_packages

setup(
    name="craftground",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        # list of dependencies e.g., 'requests', 'numpy>=1.10'
    ],
    author="yhs0602",
    author_email="jourhyang123@gmail.com",
    description="Lightweight Minecraft Environment for Reinforcement Learning",
    license="MIT",
    keywords="minecraft, reinforcement learning, environment",
    url="https://github.com/yhs0602/CraftGround",
)
