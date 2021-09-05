import sys, os
def append_root_path(path=None):
    current_path = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.dirname(current_path)
    sys.path.append(SRC_ROOT)
    if path: sys.path.append(path)
append_root_path()