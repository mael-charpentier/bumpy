import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import runpy
import numpy
import bumpy
import io
import contextlib
import types

import matplotlib.pyplot as plt

def fake_input(prompt=""):
    raise RuntimeError("input() is not allowed during automated testing")

printed = []

def fake_print(*args, **kwargs):
    for arg in args:
        printed.append(arg)

def fake_show(*args, **kwargs):
    pass
plt.show = fake_show  # 🔥 disable figures

def run_and_capture(script, numpy_lib):
    buffer = io.StringIO()

    globals_dict = {
        "__builtins__": __builtins__,
        "print": fake_print,
        "input": fake_input,  # block interactive code
    }


    with contextlib.redirect_stdout(buffer):
        sys.modules["numpy"] = numpy_lib
        sys.modules["matplotlib.pyplot"] = plt # to not show figures
        result_globals = runpy.run_path(script, globals_dict)

    return buffer.getvalue(), result_globals


EXCLUDE_TYPES = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
    types.ModuleType,
)

def filter_variables(globals_dict):
    return {
        k: v
        for k, v in globals_dict.items()
        if not k.startswith("__")
        and not isinstance(v, EXCLUDE_TYPES)
    }
def filter_variables_list(globals_list):
    return [
        v
        for v in globals_list
        if not isinstance(v, EXCLUDE_TYPES)
    ]

folder = "./fichiers_python_1736546799"

for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)

    if not file_path.endswith(".py"):
        continue


    printed = []
    out_np, globals_np = run_and_capture(file_path, numpy)
    printed_np = filter_variables_list(printed)
    try:
        printed = []
        out_bp, globals_bp = run_and_capture(file_path, bumpy)
        printed_bp = filter_variables_list(printed)
    except Exception as e:
        print(f"error in {file_name} with the implementation:", e)
        continue
        
    # Compare printed output
    assert out_bp == out_np, (
        f"Output mismatch in {file_name}\n"
        f"NumPy:\n{out_np}\n"
        f"Bumpy:\n{out_bp}"
    )
    
    for i in range(len(printed_bp)):
        assert printed_bp[i] == printed_np[i], (
            f"Printed mismatch in {file_name}, index {i}\n"
            f"NumPy:\n{printed_bp[i]}\n"
            f"Bumpy:\n{printed_np[i]}"
        )

    # Compare variables
    vars_np = filter_variables(globals_np)
    vars_bp = filter_variables(globals_bp)

    for k in vars_bp:
        assert k in vars_np, f"Missing variable {k} in bumpy ({file_name})"
        assert vars_bp[k] == vars_np[k], (
            f"Variable {k} mismatch in {file_name}\n"
            f"numpy={vars_np[k]} bumpy={vars_bp[k]}"
        )
    print(f"OK {file_name}")
