# Instructions 

Setting up a run virtual environment on a slurm cluster (EPFL SCITAS): 

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
pip install 
wandb login
```

If you want to see metrics, you will need to place an API key from weights and biases into the prompt. Alternatively select the prompt that allows you to not log metrics. 

