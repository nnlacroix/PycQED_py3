# This file initializes all the node modules in analysis_v3.

# pipeline_analysis.py must be initialized first in order to create the global
# variable search_modules, a set with all modules in analysis_v3 that contain
# node functions. Each of these modules will populate search_modules with
# the functions inside the module.

from pycqed.analysis_v3 import pipeline_analysis as pla
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import fitting as fit_module
from pycqed.analysis_v3 import plotting as plot_module
from pycqed.analysis_v3 import saving as save_module
from pycqed.analysis_v3 import data_processing as dat_proc_mod
from pycqed.analysis_v3 import ramsey_analysis as ramsey_ana
from pycqed.analysis_v3 import randomized_benchmarking_analysis as rb_ana
from pycqed.analysis_v3 import processing_pipeline as ppmod


from importlib import reload
def reload_anav3():
    reload(pla)
    reload(hlp_mod)
    reload(fit_module)
    reload(plot_module)
    reload(save_module)
    reload(dat_proc_mod)
    reload(ramsey_ana)
    reload(rb_ana)
    reload(ppmod)