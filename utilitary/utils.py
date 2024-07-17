from collections import OrderedDict

import os
import re
import struct
import sys
from colorama import Fore, Back, Style, Cursor
import numpy
import time
from datetime import datetime, timezone
from config import Config

try:
    import fcntl
except:
    print('fnctl could not be imported')

try:
    import termios
except:
    print('termios could not be imported')

#from colorama demo 6: https://github.com/tartley/colorama/blob/master/demos/demo06.py
# Fore, Back and Style are convenience classes for the constant ANSI strings that set
#     the foreground, background and style. The don't have any magic of their own.
FORES = [ Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE ]
BACKS = [ Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE ]
STYLES = [ Style.DIM, Style.NORMAL, Style.BRIGHT ]


def print_execution_time(first_stamp):
    second_stamp = int(round(time.time() * 1000))
    print('Finished: ' + datetime.now(tz=timezone.utc).isoformat())

    # Calculate the time taken in milliseconds
    time_taken = second_stamp - first_stamp

    # To get time in seconds:
    time_taken_seconds = round(time_taken / 1000)
    print(f'{time_taken_seconds} seconds or {time_taken} milliseconds')


def get_avail_gpu():
    '''
    works for linux
    '''
    result = os.popen("nvidia-smi").readlines()

    try:
    # get Processes Line
        for i in range(len(result)):
            if 'Processes' in result[i]:
                process_idx = i

        # get # of gpus
        num_gpu = 0
        for i in range(process_idx+1):
            if 'MiB' in result[i]:
                num_gpu += 1
        gpu_list = list(range(num_gpu))

        # dedect which one is busy
        for i in range(process_idx, len(result)):
            if result[i][22] == 'C':
                gpu_list.remove(int(result[i][5]))

        return (gpu_list[0])
    except:
        print('no gpu available, return 0')
        return 0

#https://stackoverflow.com/a/69582478/1694701
def get_cursor_pos():
    if(sys.platform == "win32"):
        OldStdinMode = ctypes.wintypes.DWORD()
        OldStdoutMode = ctypes.wintypes.DWORD()
        kernel32 = ctypes.windll.kernel32
        kernel32.GetConsoleMode(kernel32.GetStdHandle(-10), ctypes.byref(OldStdinMode))
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 0)
        kernel32.GetConsoleMode(kernel32.GetStdHandle(-11), ctypes.byref(OldStdoutMode))
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    else:
        OldStdinMode = termios.tcgetattr(sys.stdin)
        _ = termios.tcgetattr(sys.stdin)
        _[3] = _[3] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, _)
    try:
        _ = ""
        sys.stdout.write("\x1b[6n")
        sys.stdout.flush()

        while not _.endswith('R'):
            # Changed for manitaining compatibility between Python 3.7 and 3.8 versions
            # (:=  not supported in 3.7)
            _ += sys.stdin.read(1)
            

        res = re.match(r".*\[(?P<y>\d*);(?P<x>\d*)R", _)
    finally:
        if(sys.platform == "win32"):
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), OldStdinMode)
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), OldStdoutMode)
        else:
            termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, OldStdinMode)
    if(res):
        return (res.group("x"), res.group("y"))
    return (-1, -1)

#autogenerada por copilot y funciona, WOW!
def get_terminal_size():    
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:            
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (env['LINES'], env['COLUMNS'])
        except:
            cr = (25, 80)
    return int(cr[1]), int(cr[0])

def print_at_xy(x, y, str):
    print(f'{Cursor.POS(x, y)}{str}', end='')

#Fix state diccionary for models that are trained in more than one GPU
#by removing "module" from de beginning of each key
def fix_state_dictionary(checkpoint):
    '''
    Fix state diccionary for models that are trained in more than one GPU
    by removing "module" from de beginning of each key
    '''
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('module.'):
            new_state_dict[key[7:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict
    
def cmap(faces):
    '''
    faces: list of faces to be colored according to the color map in the config file
    Returns: list of colors corresponding to the faces
    '''
    faces = numpy.ndarray.flatten(numpy.array(faces))
    colors = [] 
    for c in range(len(faces)):
        colors.append(Config.colors[faces[c]][1])
    return colors
