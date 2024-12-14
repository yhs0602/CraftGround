from typing import Optional
from .environment import ActionSpaceVersion, CraftGroundEnvironment
from .initial_environment_config import InitialEnvironmentConfig


def make(
    initial_env_config: Optional[InitialEnvironmentConfig] = None,
    verbose=False,
    env_path=None,
    port=8000,
    action_space_version=ActionSpaceVersion.V1_MINEDOJO,
    render_action=False,
    render_alternating_eyes=False,
    use_terminate=False,
    cleanup_world=True,
    use_vglrun=False,
    track_native_memory=False,
    ld_preload=None,
    native_debug: bool = False,
    verbose_python=False,
    verbose_jvm=False,
    verbose_gradle=False,
) -> CraftGroundEnvironment:
    if not initial_env_config:
        initial_env_config = InitialEnvironmentConfig()
    return CraftGroundEnvironment(
        initial_env_config,
        verbose=verbose,
        env_path=env_path,
        port=port,
        action_space_version=action_space_version,
        render_action=render_action,
        render_alternating_eyes=render_alternating_eyes,
        use_terminate=use_terminate,
        cleanup_world=cleanup_world,
        use_vglrun=use_vglrun,
        track_native_memory=track_native_memory,
        ld_preload=ld_preload,
        native_debug=native_debug,
        verbose_python=verbose_python,
        verbose_jvm=verbose_jvm,
        verbose_gradle=verbose_gradle,
    )
