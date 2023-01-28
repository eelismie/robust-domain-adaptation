# Instructions 

Setting up a run virtual environment on a slurm cluster (EPFL Scitas / Izar): 

```
module load gcc/8.4.0-cuda
module load python/3.7.7
python3.7 -m venv ~/environments/run_env
source ~/environments/run_env/bin_activate
pip install --no-cache-dir --user -r requirements.txt
pip install -i https://test.pypi.org/simple/ tllib==0.4
```

The project made use of weights and biases for logging experiments. Regardless if you want to see metrics, run: 

```
pip install wandb
wandb login
```

If you want to log metrics, you will need to place an API key from weights and biases into the prompt. 

## Reproducing Results 

Check the [demo](notebooks/demo.ipynb) on how to select hyperparameters, and then run the [experiments](experiment_registry) with their default arguments. The datasets will be downloaded automatically, and everything should work assuming you're running the code on a slurm login node from the base directory of this repo. Important: make sure to change the path names in the experiment.yaml files to match your environment. 

To reproduce the best runs, run CFOL on MDD and CDAN with eta = 10e-6, with CFOL epoch = 10 (Visda). On DomainNet run for 10 total training epochs with LCVAR with alpha set to 0.8.

## Limitations of the Method 

Using robust risk instead of regular classification risk can only have marginal improvements on the worst class accuracy. If we use class conditioned sampling with CFOL we can get improvements in the target domain, but the small improvement may not be worth the additional effort.
