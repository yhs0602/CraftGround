from .craftground import CraftGroundEnvironment
from .initial_environment import InitialEnvironment


def make(
    verbose=False,
    env_path=None,
    port=8000,
    render_action=False,
    render_alternating_eyes=False,
    use_terminate=False,
    cleanup_world=True,
    use_vglrun=False,
    **kwargs,
) -> CraftGroundEnvironment:
    env = InitialEnvironment(**kwargs)
    return CraftGroundEnvironment(
        env,
        verbose=verbose,
        env_path=env_path,
        port=port,
        render_action=render_action,
        render_alternating_eyes=render_alternating_eyes,
        use_terminate=use_terminate,
        cleanup_world=cleanup_world,
        use_vglrun=use_vglrun,
    )
