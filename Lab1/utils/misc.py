import os

def get_results_dir(results_path, dir_name):
    results_dir = os.path.join(results_path, dir_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir