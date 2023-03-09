import tensorflow as tf
import argparse
import wandb

from model import SimpleModel
from dataset import SimpleDataset

# configure gpu devices
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# create argument parser
parser = argparse.ArgumentParser(description="wandb demo")
parser.add_argument("--num_gpus", type=int, required=True)

# data arguments
parser.add_argument("--a", type=float, required=True)
parser.add_argument("--b", type=float, required=True)
parser.add_argument("--c", type=float, required=True)
parser.add_argument("--noise", type=float, default=0)

# training arguments
parser.add_argument("--steps", type=int, default=16384)
parser.add_argument("--log_save_step", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.1)

# logging arguments
parser.add_argument("--wandb_user", type=str, required=True)
args = parser.parse_args()

# initialize wandb
wandb.init(
    project="wandb-demo",
    entity=args.wandb_user,
    config=args,
    group=f"{args.a}x + {args.b}y + {args.c} > 0",
    job_type="train",
    tags=["SimpleDataset", "SimpleModel"],
    name=f"Noise: {args.noise}, Steps: {args.steps}; LR: {args.learning_rate}",
)

# create dataset
dataset = SimpleDataset(args.a, args.b, args.c, args.noise).tfdataset(
    batch_size=args.batch_size
)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.OFF
)
dataset = dataset.with_options(options)

# set up distributed training
if args.num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = SimpleModel(
            args.batch_size, args.learning_rate, args.num_gpus, strategy
        )
else:
    model = SimpleModel(args.batch_size, args.num_gpus)

# train the model
model.train(dataset, args.steps, args.log_save_step)

a_hat = model.model.weights[0][0][0].numpy()
b_hat = model.model.weights[0][1][0].numpy()
c_hat = model.model.bias[0].numpy()

t_s = ">" if args.b > 0 else "<"
l_s = ">" if b_hat > 0 else "<"

print(f"Target: y {t_s} {-1 * args.a / args.b}x + {-1 * args.c / args.b}")
print(f"Learned: y {l_s} {-1 * a_hat / b_hat}x + {-1 * c_hat / b_hat}")
