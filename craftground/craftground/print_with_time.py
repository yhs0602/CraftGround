from datetime import datetime

do_print_with_time = False


def print_with_time(*args, **kwargs):
    if not do_print_with_time:
        return
    time_str = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{time_str}]", *args, **kwargs)
