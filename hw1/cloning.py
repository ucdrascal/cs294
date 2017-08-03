import argparse
import numpy as np
import h5py
from sklearn.linear_model import LinearRegression, Ridge
import gym
import load_policy
import tensorflow as tf
import tf_util


training_file = 'data.h5'


class EstimatorPolicy(object):
    """Wraps a fitted scikit-learn estimator to call it directly."""

    def __init__(self, actions, observations, expert_policy, use_past_obs=False,
                 use_past_act=False):
        X_train = observations[1:]

        self.use_past_obs = use_past_obs
        if use_past_obs:
            X_train = np.hstack([X_train, observations[:-1]])

        self.use_past_act = use_past_act
        if use_past_act:
            X_train = np.hstack([X_train, actions[:-1]])

        self.estimator = Ridge(alpha=0.5)
        self.estimator.fit(X_train, actions[1:])

        self.nact, self.nfeat = self.estimator.coef_.shape
        self.clear()

        self.expert_policy = expert_policy

    def __call__(self, obs):
        x = obs
        if self.use_past_obs:
            x = np.hstack([x, self.past_obs])
        if self.past_act is None:
            self.past_act = self.expert_policy(self.past_obs[None, :]).squeeze()
        if self.use_past_act:
            x = np.hstack([x, self.past_act])
        x = np.atleast_2d(x)
        act = self.estimator.predict(x)
        self.past_act = act.squeeze()
        self.past_obs = obs.squeeze()
        return act

    def clear(self):
        self.past_act = None
        n = self.nfeat
        if self.use_past_act:
            n -= self.nact
        if self.use_past_obs:
            n = n // 2
        self.past_obs = np.zeros(n)


def train(envname):
    with h5py.File(training_file, 'r') as f:
        actions = f.get('/{}/actions'.format(envname))[:]
        observations = f.get('/{}/observations'.format(envname))[:]

    actions = actions.squeeze()

    exp_pol = load_policy.load_policy('experts/{}.pkl'.format(envname))
    return EstimatorPolicy(actions, observations, exp_pol, use_past_obs=True,
                           use_past_act=True)


def run_episode(env, policy, render=False):
    obs = env.reset()
    observations = []
    rewards = []
    rewards = []
    done = False
    totalr = 0.0
    steps = 0
    while not done:
        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        totalr += reward
        steps += 1

        if render:
            env.render()

    return totalr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname',
        help='Name of the OpenAI Gym environment')
    parser.add_argument('-r', '--render', action='store_true', default=False,
        help='Render environment in a window.')
    parser.add_argument('-n', '--num_rollouts', type=int, default=10,
        help='Number of times to run the policy')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        policy = train(args.envname)
        for i in range(10):
            policy.clear()
            r = run_episode(env, policy, render=args.render)
            print("reward: {}".format(r))
