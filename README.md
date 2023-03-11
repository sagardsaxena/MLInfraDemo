# ML Infra Demo

## Set Up The Repository

```bash
# clone the repo
git clone git@github.com:sagardsaxena/MLInfraDemo.git
cd MLInfraDemo

# add python module file
module add Python3/3.10.4
bash setup.sh
```

## Start a GPU Instance on the Cluster

```bash
# create a screen
tmux new -s demo
tmux attach -t demo

# connect to a node with 2 gpus 
srun --pty --qos=medium --time=01:00:00 --mem=20gb \
 --gres=gpu:2 --cpus-per-task=8 bash

# add module files
module add cuda/11.3.1 
module add cudnn/v8.2.1
```

## Run the Demo

```bash
# activate the virtual environment
source venv/bin/activate

# run the train.py code - change the wandb username below
python3 train.py --a 1 --b 1 --c 0 --noise 0 --wandb_user your_username \
 --num_gpus 2 --batch_size 64 --steps 16384 --log_save_step 256
```
