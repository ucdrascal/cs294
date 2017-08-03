# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

The only file that you need to look at is `run_expert.py`, which is code to
load up an expert policy, run a specified number of roll-outs, and save out
data.

The `run_all.bash` script will run all environments with the expert policy and
collect all of the data in an HDF5 file `data.h5`. The `cloning.py` file has
some code to load this data, train a simple behavioral cloning model, then run
the cloned policy.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.
