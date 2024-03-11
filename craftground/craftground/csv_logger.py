from datetime import datetime


class CsvLogger:
    def __init__(self, filename, enabled: bool = False):
        self.filename = filename
        self.file = open(filename, "w")
        self.file.write("time,message\n")
        self.enabled = enabled

    def log(self, message):
        if not self.enabled:
            return
        time_str = datetime.now().strftime("%H:%M:%S.%f")
        self.file.write(f"{time_str},{message}\n")
        self.file.flush()

    def close(self):
        self.file.close()
