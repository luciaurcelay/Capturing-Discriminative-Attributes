import os

def join_path(base_path, *path_names):
    return os.path.join(base_path, *path_names)

def create_folder(path, folder_name):
    path = join_path(path, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_num_subfolders(path):
    return sum(os.path.isdir(join_path(path, elem)) for elem in os.listdir(path))


def create_new_experiment_folder(args, path):
    if args.new_experiment == 'True':
        num_experiments = get_num_subfolders(path)
        experiment_path = create_folder(path, f'experiment_{num_experiments+1}')
        return experiment_path
    else:
        return None