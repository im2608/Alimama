
import sys
import redis
import datetime
import time
runningPath = sys.path[0]

ISOTIMEFORMAT="%Y-%m-%d %X"
def getCurrentTime():
    return time.strftime(ISOTIMEFORMAT, time.localtime())
