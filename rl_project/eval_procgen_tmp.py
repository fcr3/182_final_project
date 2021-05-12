from collections import deque
import argparse
import os
import time
import torch
import numpy as np

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

import tensorflow as tf

from models.tmp_init import TMPNet_template_init
from agents.ppo import PPO

import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    parser.add_argument('model_file', type=str, default=None)

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env-name', type=str, default='fruitbot')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=0)
    parser.add_argument('--start-level', type=int, default=1000)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--exp-name', type=str, default='trial01')
    parser.add_argument('--log-dir', type=str, default='./logs_tmp')
    parser.add_argument('--method-label', type=str, default='vanilla')


    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--max_resets', type=int, default=100)

    # TMP parameters.
    parser.add_argument('--proc_size', type=int, default=3)
    parser.add_argument('--proc_strd', type=int, default=2)
    
    return parser.parse_args()


def create_venv(config):
    venv = ProcgenEnv(
        num_envs=config.num_envs,
        env_name=config.env_name,
        num_levels=config.num_levels,
        start_level=config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=config.num_threads,
    )

    print('creating venv with sl', config.start_level)

    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def rollout_one_step(agent, env, obs, steps, env_max_steps=1000):

    # Step once.
    action = agent.batch_act(obs)
    new_obs, reward, done, infos = env.step(action)
    steps += 1
    reset = steps == env_max_steps

    # Save experience.
    agent.batch_observe(
        batch_obs=new_obs,
        batch_reward=reward,
        batch_done=done,
        batch_reset=reset,
    )

    # Get rollout statistics.
    epinfo = []
    for info in infos:
        maybe_epinfo = info.get('episode')
        if maybe_epinfo:
            epinfo.append(maybe_epinfo)

    return new_obs, reward, steps, done, epinfo


def eval_model(config, agent, eval_env, model_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_log_dir = model_dir + '/' + current_time + '/eval'
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    assert(config.model_file is not None)
    agent.model.load_from_file(config.model_file)
    logger.info('Loaded model from {}.'.format(config.model_file))

    eval_epinfo_buf = deque(maxlen=100)
    eval_obs = eval_env.reset()
    eval_steps = np.zeros(config.num_envs, dtype=int)
    eval_eprewards = np.zeros(config.num_envs, dtype=float)

    nbatch = config.num_envs * config.nsteps
    # Due to some bug-like code in baseline.ppo2,
    # (and I modified PFRL accordingly) the true batch size is
    # nbatch // config.batch_size.
    n_ops_per_update = nbatch * config.nepochs / (nbatch // config.batch_size)
    max_resets = config.max_resets

    total_lengths = 0.
    total_rewards = 0.

    tstart = time.perf_counter()

    reset_cnt = 0

    while reset_cnt < max_resets:

        # Roll-out in the eval environments.
        with agent.eval_mode():
            assert not agent.training
            eval_obs, reward, steps, done, eval_epinfo = rollout_one_step(
                agent=agent,
                env=eval_env,
                obs=eval_obs,
                steps=eval_steps
            )

            next_steps = np.copy(steps)
            next_steps[done] = 0

            eval_eprewards += reward
            reset_cnt += np.sum(next_steps == 0)
            total_rewards += np.sum(eval_eprewards[next_steps == 0])

            finished_rewards = eval_eprewards[next_steps == 0]

            if len(finished_rewards) > 0:
              for fr in finished_rewards:
                logger.logkv('test_run', fr)

              logger.dumpkvs()

            eval_eprewards[next_steps == 0] = 0

            total_lengths += np.sum(eval_steps[next_steps == 0])

            eval_steps[next_steps == 0] = 0

            eval_steps = next_steps

            eval_epinfo_buf.extend(eval_epinfo)

    logger.info('eval average episode reward', total_rewards / reset_cnt)
    logger.info('eval average episode length', total_lengths / reset_cnt)

    # Save the final model.
    logger.info('Eval done.')

def run():
    configs = parse_args()

    # Configure logger.
    log_dir = os.path.join(
        configs.log_dir,
        configs.env_name,
        'nlev_{}_{}'.format(configs.num_levels, configs.distribution_mode),
        configs.method_label,
        configs.exp_name,
    )

    print('configuring logger at dir', log_dir)

    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])

    # Create venvs.
    eval_venv = create_venv(configs)

    # Create policy.
    tmpnet = TMPNet_template_init

    policy = tmpnet(
        obs_space=eval_venv.observation_space,
        num_outputs=eval_venv.action_space.n,
        proc_conv_ksize=configs.proc_size,
        proc_conv_stride=configs.proc_strd,
        log_dir=log_dir,
        gpu=configs.gpu
    )

    # Create agent and train.
    optimizer = torch.optim.Adam(policy.parameters(), lr=configs.lr, eps=1e-5)
    ppo_agent = PPO(
        model=policy,
        optimizer=optimizer,
        gpu=configs.gpu,
        gamma=configs.gamma,
        lambd=configs.lam,
        value_func_coef=configs.vf_coef,
        entropy_coef=configs.ent_coef,
        update_interval=configs.nsteps * configs.num_envs,
        minibatch_size=configs.batch_size,
        epochs=configs.nepochs,
        clip_eps=configs.clip_range,
        clip_eps_vf=configs.clip_range,
        max_grad_norm=configs.max_grad_norm,
    )
    eval_model(configs, ppo_agent, eval_venv, log_dir)


if __name__ == '__main__':
    run()
