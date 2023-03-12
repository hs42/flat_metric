import os
import sys
import importlib


def dynamic_import(filepath, class_name):
    tail, head = os.path.split(filepath)
    directory_of_this_file = os.path.dirname(os.path.abspath(__file__))
    tail_corrected = os.path.join(directory_of_this_file, os.pardir, os.pardir, tail)
    sys.path.append(tail_corrected)
    module = importlib.import_module(".".join(head.split(".")[:-1]))
    sys.path.remove(tail_corrected)

    my_class = getattr(module, class_name)

    return my_class
