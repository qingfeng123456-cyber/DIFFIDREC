
import datetime
import os


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_names = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        color_index = color_names.index(color)
    except:
        color_index = len(color_names) - 1
    color_prefix = "\033["
    if highlight:
        color_prefix += "1;3"
    else:
        color_prefix += "0;3"
    color_prefix += str(color_index) + "m"
    return color_prefix + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%b-%d-%Y_%H-%M-%S")

    return current_time

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
