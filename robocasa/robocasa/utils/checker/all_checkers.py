import sys
from robocasa.utils.checker import *

# aggregate all checker functions into a string
current_module = sys.modules[__name__]
function_names = [
    name for name in dir(current_module)
    if callable(getattr(current_module, name)) and name.endswith("_checker")
]

ALL_CHECKERS = ", ".join(function_names)

if __name__ == '__main__':
    print(ALL_CHECKERS)