import os
import yaml
import json

"""
Script for generating a large amount of batch jobs at once 
"""

base_path = "/home/mielonen/robust-domain-adaptation"

base_experiments = [
    "experiment_registry/mcc-pacs.yaml", 
    "experiment_registry/mcc-visda2017.yaml",
    "experiment_registry/mdd-visda2017.yaml"
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
    For each base experiment set seeds to [1, 2, 3], Cfol to true, and launch everything in the cluster. 
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
                    assert(gamma >= 0)
                    assert(gamma <= 0)
                    newDict["experiment"]["global_params"]["gamma"] = gamma
                    yield newDict

        self.generator = new(self.generator)
        return self

    def setCfol(self, val):
    
        def new(base):
            for dict_ in base:
                newDict = dict_
                newDict["experiment"]["global_params"]["cfol_sampling"] = val
                yield newDict

        self.generator = new(self.generator)
        return self

    def launch(self):
        a = self.generator
        for i in a:
            jsonString = json.dumps(i)
            str_cmd = f"sbatch submit_job.run '{jsonString}'"
            os.system(str_cmd)

experimentBuilder([base_experiments[0]]).setSeed([42]).setCfol(True).launch()