import numpy as np


def bootstrap_func(function, *args, n_iter=10):
    results = []
    current_args = args
    for i in range(n_iter):
        choice = np.random.choice(len(current_args[0]), len(current_args[0]))
        # choice = np.arange(len(current_args[0]))
        new_args = [arg[choice] for arg in current_args]
        results.append(function(*new_args))
    # if there are more than one output of results, then do average on all
    avg = []
    std = []
    for i in range(len(results[0])):
        res_dim_i = [result[i] for result in results]
        res_dim_i = np.array(res_dim_i)
        # print("res shape:", res_dim_i.shape)
        avg.append(np.average(res_dim_i, axis=0))
        std.append(np.std(res_dim_i, axis=0))
    return avg, std
