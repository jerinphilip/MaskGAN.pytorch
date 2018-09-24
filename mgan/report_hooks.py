from pprint import pprint
import sys

class DevNull:
    def __call__(self, _dict):
        pass

class Log:
    def __init__(self, log_file=sys.stderr):
        self.log_file = log_file

    def __call__(self, _dict):
        pprint(_dict, self.log_file)
