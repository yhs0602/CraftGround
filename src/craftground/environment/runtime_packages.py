import importlib.resources

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
