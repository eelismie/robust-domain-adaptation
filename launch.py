import os

base_path = "/home/mielonen/robust-domain-adaptation"

experiment_list = [
    "experiment_registry/daln-pacs-c.yaml",
    "experiment_registry/mcc-pacs-c.yaml",
    "experiment_registry/mdd-pacs-c.yaml",
    "experiment_registry/daln-pacs.yaml",
    "experiment_registry/mcc-pacs.yaml",
    "experiment_registry/mdd-pacs.yaml",
    "experiment_registry/daln-visda2017.yaml",
    "experiment_registry/mcc-visda2017.yaml",
    "experiment_registry/mdd-visda2017.yaml"
]

def modify(yaml, key, value):
    """
    TODO : take dict in yaml and change value in key to value
    """
    pass 

for experiment_path in experiment_list:
    full_path = os.path.join(base_path, experiment_path)
    str_cmd = f"sbatch submit_job.run '{full_path}' "
    os.system(str_cmd)
