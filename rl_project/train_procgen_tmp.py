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
from models.tmpv1 import TMPNet1
from models.tmpv2 import TMPNet2
from models.tmpv3 import TMPNet3
from agents.ppo import PPO

import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

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
    parser.add_argument("--kernel-norm-diff", default=False, action="store_true")
    parser.add_argument("--kernel-norm", default=False, action="store_true")
    parser.add_argument("--impala-layer-init", type=int, default=0)
    parser.add_argument("--init-style", type=str, default='resize')
    parser.add_argument('--init-all-input_channels', default=False, 
                        action="store_true")


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
    parser.add_argument('--max-steps', type=int, default=25_000_000)
    parser.add_argument('--save-interval', type=int, default=100)

    # TMP parameters.
    parser.add_argument('--TMPv', type=str, default='init', 
                        choices=['v1', 'v2', 'v3', 'init'])
    parser.add_argument('--grad', type=lambda x : bool(x), default=False)
    parser.add_argument('--proc-size', type=int, default=3)
    parser.add_argument('--proc-strd', type=int, default=2)
    parser.add_argument('--model-file', type=str, default=None)
    parser.add_argument('--target-width', type=int, default=7)
    parser.add_argument('--pooling', type=int, default=2)
    parser.add_argument('--out-feats', type=int, default=256)
    parser.add_argument('--conv-out-feats', type=int, default=32)
    
    return parser.parse_args()


def create_venv(config, valid=False):
    venv = ProcgenEnv(
        num_envs=config.num_envs,
        env_name=config.env_name,
        num_levels=0 if valid else config.num_levels,
        start_level=0 if valid else config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=config.num_threads,
    )
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
    steps[done] = 0

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

    return new_obs, steps, epinfo


def train(config, agent, train_env, test_env, model_dir, 
          policy, log_kernel_norm_diff, log_kernel_norm, 
          init_all_input_channels):

  
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = model_dir + '/' + current_time + '/train'
    test_log_dir = model_dir + '/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    if config.model_file is not None:
        agent.model.load_from_file(config.model_file)
        logger.info('Loaded model from {}.'.format(config.model_file))
    else:
        logger.info('Train agent from scratch.')

    train_epinfo_buf = deque(maxlen=100)
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    test_epinfo_buf = deque(maxlen=100)
    test_obs = test_env.reset()
    test_steps = np.zeros(config.num_envs, dtype=int)

    nbatch = config.num_envs * config.nsteps
    # Due to some bug-like code in baseline.ppo2,
    # (and I modified PFRL accordingly) the true batch size is
    # nbatch // config.batch_size.
    n_ops_per_update = nbatch * config.nepochs / (nbatch // config.batch_size)
    nupdates = config.max_steps // nbatch
    max_steps = config.max_steps // config.num_envs

    logger.info('Start training for {} steps (approximately {} updates)'.format(
        config.max_steps, nupdates))

    tstart = time.perf_counter()
    for step_cnt in range(max_steps):

        # Roll-out in the training environments.
        assert agent.training
        train_obs, train_steps, train_epinfo = rollout_one_step(
            agent=agent,
            env=train_env,
            obs=train_obs,
            steps=train_steps
        )
        train_epinfo_buf.extend(train_epinfo)

        # Roll-out in the test environments.
        with agent.eval_mode():
            assert not agent.training
            test_obs, test_steps, test_epinfo = rollout_one_step(
                agent=agent,
                env=test_env,
                obs=test_obs,
                steps=test_steps
            )
            test_epinfo_buf.extend(test_epinfo)

        assert agent.training
        num_ppo_updates = agent.n_updates // n_ops_per_update

        if (step_cnt + 1) % config.nsteps == 0:
            tnow = time.perf_counter()
            fps = int(nbatch / (tnow - tstart))

            with train_summary_writer.as_default():
              tf.summary.scalar('eprewmean', safe_mean([info['r'] for info in train_epinfo_buf]), step=step_cnt+1)
              tf.summary.scalar('eplenmean', safe_mean([info['l'] for info in train_epinfo_buf]), step=step_cnt+1)
              tf.summary.scalar('total_steps', (step_cnt + 1) * config.num_envs, step=step_cnt+1)
              tf.summary.scalar('fps', fps, step=step_cnt+1)
              tf.summary.scalar('num_ppo_updates', num_ppo_updates, step=step_cnt+1)
              train_stats = agent.get_statistics()
              for stats in train_stats:
                logger.logkv(stats[0], stats[1])
                tf.summary.scalar(stats[0], stats[1], step=step_cnt+1)

            with test_summary_writer.as_default():
              tf.summary.scalar('eval_eprewmean', safe_mean([info['r'] for info in test_epinfo_buf]), step=step_cnt+1)
              tf.summary.scalar('eval_eplenmean', safe_mean([info['l'] for info in test_epinfo_buf]), step=step_cnt+1)
            
            logger.logkv('steps', step_cnt + 1)
            logger.logkv('total_steps', (step_cnt + 1) * config.num_envs)
            logger.logkv('fps', fps)
            logger.logkv('num_ppo_update', num_ppo_updates)
            logger.logkv('eprewmean',
                        safe_mean([info['r'] for info in train_epinfo_buf]))
            logger.logkv('eplenmean',
                        safe_mean([info['l'] for info in train_epinfo_buf]))
            logger.logkv('eval_eprewmean',
                        safe_mean([info['r'] for info in test_epinfo_buf]))
            logger.logkv('eval_eplenmean',
                        safe_mean([info['l'] for info in test_epinfo_buf]))

            if log_kernel_norm_diff:
              norm_diffs = policy.kernel_diff_norm()
              for k, v in norm_diffs.items():
                logger.logkv('avg_frob_kernel_diff_layer' + str(k), v)

            if log_kernel_norm:
              norms = policy.kernel_norm()
              for k, v in norms.items():
                logger.logkv('avg_frob_kernel_layer' + str(k), v)
            
            logger.dumpkvs()

            if num_ppo_updates % config.save_interval == 0:
                model_path = os.path.join(
                    model_dir, 'model_{}.pt'.format(num_ppo_updates + 1))
                agent.model.save_to_file(model_path)
                logger.info('Model save to {}'.format(model_path))

            tstart = time.perf_counter()

    # Save the final model.
    logger.info('Training done.')
    model_path = os.path.join(model_dir, 'model_final.pt')
    agent.model.save_to_file(model_path)
    logger.info('Model save to {}'.format(model_path))


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
    train_venv = create_venv(configs, valid=False)
    valid_venv = create_venv(configs, valid=True)

    # Create policy.
    tmpnet = TMPNet_template_init
    if configs.TMPv == 'v1':
        tmpnet = TMPNet1
    if configs.TMPv == 'v2':
        tmpnet = TMPNet2
    if configs.TMPv == 'v3':
        tmpnet = TMPNet3

    policy = tmpnet(
        obs_space=train_venv.observation_space,
        num_outputs=train_venv.action_space.n,
        out_features=configs.out_feats,
        conv_out_features=configs.conv_out_feats,
        proc_conv_ksize=configs.proc_size,
        proc_conv_stride=configs.proc_strd,
        pooling=configs.pooling,
        target_width=configs.target_width,
        impala_layer_init=configs.impala_layer_init,
        init_style=configs.init_style,
        log_dir=log_dir,
        init_all_input_channels=configs.init_all_input_channels,
        grad_on=configs.grad,
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
    train(configs, ppo_agent, train_venv, valid_venv, log_dir, policy, 
          configs.kernel_norm_diff if configs.TMPv == 'init' else False, 
          configs.kernel_norm if configs.TMPv == 'init' else False, 
          configs.init_all_input_channels if configs.TMPv == 'init' else False)


if __name__ == '__main__':
    run()
