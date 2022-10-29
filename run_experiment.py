import yaml
import inspect
import click 
from model_registry import models
from dataset_registry import datasets
from routine_registry import routines 


def run_checks(dict_): 

    """
    TODO: check the signature of each of the functions and classes in dict_ and compare them against the provided args
    """

    pass 

def load_models(models_list, glob_params):

    instantiations = []
    
    for e in models_list:
        if (type(e) == type("")):
            name = e 
            model = models.__dict__[name](opt=glob_params)
            instantiations.append((name , model))
        if (type(e) == type({"a" : 1})):
            name = list(e.keys())[0]
            values = list(e.values())[0]
            model = models.__dict__[name](opt=glob_params, **values)
            instantiations.append((name , models))

    #the returned list contains name : instance of each class 
    return { k : v for k, v in instantiations }

def load_datasets(target_dataset, source_dataset, glob_params):

    class_name_t = target_dataset["class_name"]
    class_name_s = source_dataset["class_name"]
    target = datasets.__dict__[class_name_t](target_dataset["path"], opt=glob_params)
    source = datasets.__dict__[class_name_s](source_dataset["path"], opt=glob_params)

    target_train, target_val = target.get_loaders(
        train_transform_args=target_dataset["train"]["transform_args"],
        val_transform_args=target_dataset["validate"]["transform_args"]
    )

    source_train, source_val = source.get_loaders(
        train_transform_args=source_dataset["train"]["transform_args"],
        val_transform_args=source_dataset["validate"]["transform_args"]
    )

    result = { 
        target_dataset["train"]["name"] : target_train, 
        target_dataset["validate"]["name"] : target_val,
        source_dataset["train"]["name"] : source_train, 
        source_dataset["validate"]["name"] : source_val
    }

    return result

def run_routines(routine_list, models, datasets, glob_params):

        instances = {**models, **datasets}

        for r in routine_list:
            routine_name = list(r.keys())[0]
            args = r[routine_name] 
            kwargs = { k : instances[v] for k, v in args.items()}
            kwargs = {**kwargs, **{"opt" : glob_params}}
            routine = routines.__dict__[routine_name]
            routine(**kwargs)

@click.command()
@click.argument('filename')
@click.argument('debug', default=False)
def main(filename, debug):

    dict_ = None
    with open(filename, "r") as stream:
        dict_ = yaml.safe_load(stream)

    if (debug):
        run_checks(dict_)

    experiment = dict_["experiment"]
    models_list = load_models(experiment["models"], experiment["global_params"])
    datasets_list = load_datasets(experiment["target_dataset"], experiment["source_dataset"], experiment["global_params"])
    run_routines(experiment["routines"], models_list, datasets_list, experiment["global_params"])

if __name__ == "__main__":
    main()

