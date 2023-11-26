import datetime as dt
import os


class Utils:
    def get_datetime(self):
        date = dt.datetime.now().strftime("%Y-%m-%d") 
        return date

        
    def does_file_exist(self, file_name):
        return os.path.exists(file_name)
        