from datetime import datetime


def print_with_time(*args, **kwargs):
    time_str = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{time_str}]", *args, **kwargs)
