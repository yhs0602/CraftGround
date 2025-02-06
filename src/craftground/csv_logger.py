from contextlib import contextmanager
from datetime import datetime

from enum import Enum


class LogBackend(Enum):
    NONE = 0
    CSV = 1
    CONSOLE = 2
    BOTH = 3


class CsvLogger:
    def __init__(
        self, filename, profile: bool = False, backend: LogBackend = LogBackend.NONE
    ):
        self.do_profile = profile
        self.backend = backend
        if self.do_profile and backend not in [LogBackend.CSV, LogBackend.BOTH]:
            raise ValueError("Cannot enable profiling without a csv backend")
        if backend in [LogBackend.CSV, LogBackend.BOTH]:
            self.filename = filename
            self.file = open(filename, "w")
            self.file.write("time,message\n")

    def log(self, message):
        if self.backend == LogBackend.NONE:
            return
        time_str = datetime.now().strftime("%H:%M:%S.%f")
        if self.backend in [LogBackend.CSV, LogBackend.BOTH]:
            self.file.write(f"{time_str},{message}\n")
            self.file.flush()
        if self.backend in [LogBackend.CONSOLE, LogBackend.BOTH]:
            print(f"{time_str}: {message}")

    def profile_start(self, tag):
        if not self.do_profile:
            return
        time_str = datetime.now().strftime("%H:%M:%S.%f")
        if self.backend in [LogBackend.CSV, LogBackend.BOTH]:
            self.file.write(f"{time_str}, start, {tag}\n")
            self.file.flush()
        if self.backend in [LogBackend.CONSOLE, LogBackend.BOTH]:
            print(f"{time_str}: start: {tag}")

    def profile_end(self, tag):
        if not self.do_profile:
            return
        time_str = datetime.now().strftime("%H:%M:%S.%f")
        if self.backend in [LogBackend.CONSOLE, LogBackend.BOTH]:
            print(f"{time_str}: end: {tag}")
        if self.backend in [LogBackend.CSV, LogBackend.BOTH]:
            self.file.write(f"{time_str}, end, {tag}\n")
            self.file.flush()

    def close(self):
        if self.backend in [LogBackend.CSV, LogBackend.BOTH]:
            self.file.close()

    @contextmanager
    def profile(self, tag):
        """Context manager for profiling."""
        if not self.do_profile:
            yield
            return

        time_str = datetime.now().strftime("%H:%M:%S.%f")
        if self.backend in [LogBackend.CSV, LogBackend.BOTH]:
            self.file.write(f"{time_str}, start, {tag}\n")
            self.file.flush()
        if self.backend in [LogBackend.CONSOLE, LogBackend.BOTH]:
            print(f"{time_str}: start: {tag}")

        try:
            yield
        finally:
            time_str = datetime.now().strftime("%H:%M:%S.%f")
            if self.backend in [LogBackend.CONSOLE, LogBackend.BOTH]:
                print(f"{time_str}: end: {tag}")
            if self.backend in [LogBackend.CSV, LogBackend.BOTH]:
                self.file.write(f"{time_str}, end, {tag}\n")
                self.file.flush()
