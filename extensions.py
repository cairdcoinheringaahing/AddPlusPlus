import sys
import time

def getchar():
    return sys.stdin.read(1)

getcmd = {

    'getchar': (
        0,
        getchar
    ),

    'divmod': (
        2,
        divmod
    ),

    'sleep': (
        1,
        time.sleep
    ),

}
