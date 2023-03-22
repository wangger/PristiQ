from NAS_Module import *
from NAS_Module_Qiskit import *


# TODO(Note): The key of the following dict should be consistent
layer_name_dict = {
    'v0': None,
    'v10': QLayer_10,     # adapted by Zhirui
    'v2': QLayer_2,
    'v13': QLayer_13,
    'v19': QLayer_19,
    'v5': QLayer_5,
}   # (name, layer_class)


qiskit_layer_name_dict = {
    'v0': None,
    'v10': build_QLayer_10,     # adapted by Zhirui
    'v2': build_QLayer_2,
    'v13': build_QLayer_13,
    'v19': build_QLayer_19,
    'v5': build_QLayer_5,
}   # (name, layer_class)