import os
import sys
import time

def getchar():
    return sys.stdin.read(1)

def readfile(file):
    try:
        return open(file, 'r', encoding = 'utf-8').read()
    except:
        return file

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

    'integer': (
        1,
        int
    ),

    'string': (
        1,
        str
    ),

    'array': (
        1,
        list
    ),
    
    'read': (
        1,
        readfile,
    ),

}
