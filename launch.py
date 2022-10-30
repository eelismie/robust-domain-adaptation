import os

base_path = "/home/mielonen/robust-domain-adaptation"

experiment_list = [
    "experiment_registry/pixelda-mnist-reverse.yaml",
    "experiment_registry/pixelda-mnist.yaml",
    "experiment_registry/pixelda-visda2017.yaml"
]

for experiment_path in experiment_list:
    full_path = os.path.join(base_path, experiment_path)
    str_cmd = f"sbatch submit_job.run '{full_path}' "
    os.system(str_cmd)
