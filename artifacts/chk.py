import pandas as pd
import importlib
import pkg_resources

def imp(module_name):
    try:
        actual_module = "sklearn" if module_name == "scikit-learn" else module_name
        pkg_resources.require(module_name)
        mod = importlib.import_module(actual_module)
        return getattr(mod, '__version__', 'No version attribute')
    except (ModuleNotFoundError, pkg_resources.DistributionNotFound):
        return "❌ Not Installed"
    
with open('requirements.txt', 'r') as fileob:
    ls = fileob.read().split('\n')

ls.pop()

for i in ls:
    print(f"{i}:{imp(i)}")

