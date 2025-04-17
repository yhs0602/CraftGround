def check_vglrun():
    from shutil import which

    return which("vglrun") is not None
