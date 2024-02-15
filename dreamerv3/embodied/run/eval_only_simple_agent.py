import re

import embodied
import numpy as np


def eval_only_simple_agent(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    # reward terms
    reward_terms = {k: v for k, v in ep.items() if '_reward' in k}
    logger.add({'length': length, 'score': score, **reward_terms}, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()

    for key, value in reward_terms.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      stats[f'sum_{key}'] = reward_terms[key].sum()
      stats[f'mean_{key}'] = reward_terms[key].mean()
      stats[f'max_{key}'] = reward_terms[key].max(0).mean()
      stats[f'min_{key}'] = reward_terms[key].min(0).mean()
      stats[f'std_{key}'] = reward_terms[key].std()

    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())

  def step_fn(tran, worker):
    logger.add({k: v for k, v in tran.items() if '_reward' in k}, prefix='step')
    logger.add({k: v for k, v in tran.items() if 'location' in k}, prefix='step')
    logger.add({k: v for k, v in tran.items() if 'velocity' in k}, prefix='step')
    logger.write(fps=True)
  driver.on_step(step_fn)

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    driver(policy, steps=100)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  logger.write()