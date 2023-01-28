import os
import yaml
import json

"""
Script for Slurm cluster: 
Use this for running versions of the same experiments with different hyperparameters
"""

base_path = "/home/mielonen/robust-domain-adaptation"

base_experiments = [
    "experiment_registry/mcc-domain.yaml",
    "experiment_registry/mdd-domain.yaml",
    "experiment_registry/cdan-domain.yaml",
    "experiment_registry/mcc-visda2017.yaml",
    "experiment_registry/mdd-visda2017.yaml",
    "experiment_registry/cdan-visda2017.yaml"
]

base_experiments = [os.path.join(base_path, e) for e in base_experiments]

def yamlToDict(full_path):
    with open(full_path, "r") as stream:
        dict_ = yaml.safe_load(stream)
    return dict_

class experimentBuilder:

    """
    Usage: 
    experimentBuilder(base_experiments).setSeed([1,2,3]).setCfol(True).launch() 

    Explanation:
    For each base experiment set seeds to [1, 2, 3], Cfol to true, and submit everything as a slurm batch job. 
    """

    def __init__(self, fnames):

        self.base_list = [yamlToDict(name) for name in fnames]

        def base():
            for dic in self.base_list: 
                yield dic

        self.generator = base()

    def setSeed(self, list_):

        def new(base):
            for dict_ in base:
                for seed in list_:
                    newDict = dict_
                    newDict["experiment"]["global_params"]["seed"] = seed
                    yield newDict

        self.generator = new(self.generator)
        return self


    def setGamma(self, list_):

        def new(base):
            for dict_ in base:
                for gamma in list_:
                    newDict = dict_
                    assert(gamma > 0)
                    assert(gamma < 1)
                    newDict["experiment"]["global_params"]["cfol_gamma"] = gamma
                    yield newDict

        self.generator = new(self.generator)
        return self

    def setMethod(self, val):
    
        def new(base):
            for dict_ in base:
                newDict = dict_
                newDict["experiment"]["global_params"]["reweight_method"] = val
                yield newDict

        self.generator = new(self.generator)
        return self

    def setEta(self, list_):

        def new(base):
            for dict_ in base:
                for eta in list_:
                    newDict = dict_
                    newDict["experiment"]["global_params"]["cfol_eta"] = eta
                    yield newDict

        self.generator = new(self.generator)
        return self

    def setAlpha(self, list_):

        def new(base):
            for dict_ in base:
                for alpha in list_:
                    newDict = dict_
                    newDict["experiment"]["global_params"]["cvar_alpha"] = alpha
                    yield newDict

        self.generator = new(self.generator)
        return self

    def launch(self):
        a = self.generator
        for i in a:
            jsonString = json.dumps(i)
            str_cmd = f"sbatch submit_job.run '{jsonString}'"
            os.system(str_cmd)

experimentBuilder(base_experiments).setSeed([24]).setMethod("lcvar").setAlpha([0.80]).launch()