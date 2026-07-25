import importlib.resources
import shutil
from pathlib import Path

# Build artifacts CMake/Gradle write into the runtime package's own directory
# at first run (see minecraft/mc121/MANIFEST.in's prune/exclude list, which
# keeps these out of the wheel but can't stop them from appearing later).
# A stale CMakeCache.txt pins a failed find_package() result (e.g. JNI
# NOTFOUND from a previous run without JAVA_HOME set) across later runs.
_NATIVE_BUILD_ARTIFACTS = [
    "CMakeCache.txt",
    "CMakeFiles",
    "cmake_install.cmake",
    "compile_commands.json",
    "Makefile",
    "_deps",
]

# Maps the public `mc_version` string to the installed runtime package that
# ships the corresponding Fabric mod's Gradle project (see
# minecraft/mc121/pyproject.toml, minecraft/mc262/pyproject.toml).
_RUNTIME_PACKAGES = {
    "1.21": "craftground_runtime_mc121",
    "26.2": "craftground_runtime_mc262",
}


def resolve_runtime_env_path(mc_version: str) -> str:
    package_name = _RUNTIME_PACKAGES.get(mc_version)
    if package_name is None:
        raise ValueError(
            f"Unsupported mc_version: {mc_version!r}. "
            f"Supported: {sorted(_RUNTIME_PACKAGES)}"
        )
    try:
        return str(importlib.resources.files(package_name))
    except ModuleNotFoundError as e:
        dist_name = package_name.replace("_", "-")
        raise ModuleNotFoundError(
            f"{dist_name} is not installed. Run `pip install {dist_name}` "
            f"(or the matching `craftground[...]` extra) to use mc_version={mc_version!r}."
        ) from e


def clear_native_build_cache(mc_version: str = "1.21") -> None:
    """Delete CMake/Gradle build artifacts from the installed runtime package.

    Useful after changing JAVA_HOME or upgrading the native toolchain, since a
    stale CMakeCache.txt otherwise keeps reusing a previous (possibly failed)
    find_package() result.
    """
    env_path = Path(resolve_runtime_env_path(mc_version))
    for name in _NATIVE_BUILD_ARTIFACTS:
        target = env_path / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
