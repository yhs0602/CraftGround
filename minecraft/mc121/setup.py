"""Ship the shared build inputs with the runtime so Gradle can run outside a checkout."""

from pathlib import Path
from shutil import copytree

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def copy_shared_sources(destination):
    # An unpacked sdist already carries these trees beside build.gradle.
    root = Path(__file__).resolve().parent
    if not (root / "shared-java").is_dir():
        root = root.parents[1]
    for relative in ("shared-java/src", "shared-native/gl-capture"):
        copytree(root / relative, Path(destination) / relative, dirs_exist_ok=True)


class RuntimeSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        copy_shared_sources(base_dir)


class RuntimeBuild(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [module for module in modules if module[1] != "setup"]

    def run(self):
        super().run()
        copy_shared_sources(Path(self.build_lib) / "craftground_runtime_mc121")


setup(cmdclass={"sdist": RuntimeSdist, "build_py": RuntimeBuild})
