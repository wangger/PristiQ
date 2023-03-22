controller_params = {
    "sw_space": (['v10', 'v2', 'v13', 'v19', 'v5'],  # At least have 1 layers
                 ['v0', 'v10', 'v2', 'v13', 'v19', 'v5'],
                 ['v0', 'v10', 'v2', 'v13', 'v19', 'v5'],
                 ['v0', 'v10', 'v2', 'v13', 'v19', 'v5'],
                 ['v0', 'v10', 'v2', 'v13', 'v19', 'v5']),

    # # dataflow 1, dataflow 2, PE for d1, BW for d1
    # "hw_space": (list(range(8, 50, 8)), list(range(1, 9, 1)), [32, 64], [32, 64],
    # [3], [2], [2], [2]),  # TODO: Hardware
    "hw_space": (),
    'max_episodes': 100,    # epoch for RL
    "num_children_per_episode": 1,  # sample number for each episode # TODO: BS=1
    # "num_hw_per_child": 10,  # TODO: hardware
    'hidden_units': 35,  # complexity of RNN, i.e., embedding and the classifier
}

# # TODO: Hardware
# HW_constraints = {
#     "r_Ports_BW": 1024,
#     "r_DSP": 2520,
#     "r_BRAM": 1824,
#     "r_BRAM_Size": 18000,
#     "BITWIDTH": 16,
#     "target_HW_Eff": 1
# }