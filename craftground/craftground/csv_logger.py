from datetime import datetime


class CsvLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.file.write("time,message\n")

    def log(self, message):
        time_str = datetime.now().strftime("%H:%M:%S.%f")
        self.file.write(f"{time_str},{message}\n")

    def close(self):
        self.file.close()
