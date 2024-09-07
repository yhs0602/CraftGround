from setuptools import setup, find_packages

packages = find_packages()
setup(
    name="craftground",
    version="2.4.28",
    packages=packages,
    install_requires=["gymnasium", "Pillow", "numpy", "protobuf", "typing_extensions"],
    author="yhs0602",
    author_email="jourhyang123@gmail.com",
    description="Lightweight Minecraft Environment for Reinforcement Learning",
    license="MIT",
    keywords="minecraft, reinforcement learning, environment",
    url="https://github.com/yhs0602/CraftGround",
    package_data={
        packages[0]: [
            "craftground/MinecraftEnv/*",
            "craftground/DejaVuSans-ExtraLight.ttf",
        ]
    },
    include_package_data=True,
)
