import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    # value を取り除く
    prev_latent = {k: v for k, v in prev_latent.items() if 'value' not in k}
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    for key, critic in self.task_behavior.ac.critics.items():
      latent[f'{key}_value'] = critic.net(latent).mean()
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    # 世界モデルの学習
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    context['embed'] = wm_outs['embed']
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    aux_args = {'observe': self.wm.observe}
    # actor-criticの学習
    _, mets = self.task_behavior.train(self.wm.imagine, start, context, **aux_args)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context, **aux_args)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics
  
  def pretrain_wm(self, data, state):
    self.config.jax.jit and print('Tracing pretrain_wm function.')
    metrics = {}
    data = self.preprocess(data)
    # 世界モデルの学習
    state, _, mets = self.wm.train(data, state)
    metrics.update(mets)
    outs = {}
    return outs, state, metrics
  
  def train_actor_critic(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    # 世界モデルの推論
    _, (state, wm_outs, mets) = self.wm.loss(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    context['embed'] = wm_outs['embed']
    # stop-gradient
    context = sg(context)
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    aux_args = {'observe': self.wm.observe}
    # actor-criticの学習
    _, mets = self.task_behavior.train(self.wm.imagine, start, context, **aux_args)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context, **aux_args)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)

    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})

    # imagination report
    img_report = {}
    policy = lambda s: self.task_behavior.ac.actor(sg(s)).sample(seed=nj.rng())
    img_report.update(self.wm.img_report(data, policy))
    mets = self.task_behavior.img_report(data)
    img_report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.img_report(data)
      img_report.update({f'expl_{k}': v for k, v in mets.items()})
    report.update({f'{k}_img': v for k, v in img_report.items()})

    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj
  
  def observe(self, embed, action, is_first, context, state=None):
    swap = lambda x: {k: v.transpose([1, 0] + list(range(2, len(v.shape)))) for k, v in x.items()}
    post, _ = self.rssm.observe(embed, action, is_first, state)
    obs_traj = {'action': action, **post}
    obs_traj['cont'] = (1.0 - context['is_terminal']).astype(jnp.float32)
    discount = 1 - 1 / self.config.horizon
    obs_traj['weight'] = jnp.cumprod(discount * obs_traj['cont'], 0) / discount

    return swap(obs_traj)

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6], data['action'][:6],
        data['is_first'][:6])
    start = {k: v[:, 4] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, recon[key].mode(), model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report
  
  def img_report(self, data, policy):
    swap = lambda x: {k: v.transpose([1, 0] + list(range(2, len(v.shape)))) for k, v in x.items()} 
    img_report = {}
    context, _ = self.rssm.observe(
        self.encoder(data)[:6], data['action'][:6],
        data['is_first'][:6])
    print(f'context.keys(): {context.keys()}')
    context.update({k: v[:6] for k, v in data.items() if k not in set(context.keys())})
    print(f'context.keys(): {context.keys()}')
    start = {k: v[:, 4] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    traj = self.imagine(policy, start, self.config.imag_horizon)
    traj = swap(traj)
    openl = self.heads['decoder'](traj)
    for key in self.heads['decoder'].cnn_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      video = jnp.concatenate([model], 2)
      img_report[f'openl_{key}'] = jaxutils.video_grid(video)
    return img_report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)
    self.enable_obs_actor_loss = config.enable_obs_actor_loss
    self.enable_bc = config.enable_bc
    self.random_behavior = behaviors.Random(None, act_space, config, name='random_behavior')

  def sample_dict_array(self, d: dict, n: int):
    '''
    Args: d is a dict of arrays
    Return: a dict of arrays have n samples
    '''
    swap = lambda x: {k: v.transpose([1, 0] + list(range(2, len(v.shape)))) for k, v in x.items()}
    d = swap(d)
    batch_size = next(iter((d.values()))).shape[0]
    assert n <= batch_size, f'n must be less than batch_size, but n: {n} and batch_size: {batch_size}'
    rng = jax.random.PRNGKey(0)

    # 0からnまでの整数のリストを作成
    population = jnp.arange(batch_size)
    
    # シャッフルされたリストから最初のk個の要素を選択
    shuffled_indices = jax.random.shuffle(rng, population)
    sample_indices = shuffled_indices[:n]
    

    # 選択したインデックスに基づいてサンプリング
    samples = {}
    for k in d.keys():
      samples[k] = d[k][sample_indices]

    return swap(samples)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context, **kwargs):
    if self.config.enable_combo:
      observe = kwargs.get('observe', None)

    # ポリシーの損失
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      if self.config.enable_combo and self.config.enable_combo_to_actor:
        obs_traj = observe(context['embed'], context['action'], context['is_first'], context)
        obs_batch_size = next(iter((obs_traj.values()))).shape[1]
        traj = self.sample_dict_array(traj, obs_batch_size)
        traj_dict = {'img_traj': traj, "obs_traj": obs_traj}

        if self.config.combo_random_policy:
          random_policy = lambda s: self.random_behavior.policy(None, s['deter'])[0]['action'].sample(seed=nj.rng())
          random_traj = imagine(random_policy, start, self.config.imag_horizon)
          obs_batch_size = list(traj_dict['obs_traj'].values())[0].shape[1]
          random_traj = self.sample_dict_array(random_traj, obs_batch_size)
          traj_dict['random_traj'] = random_traj

        loss, metrics = self.combo_loss(traj_dict)
        traj = traj_dict
      else:
        loss, metrics = self.loss(traj)
      if self.config.enable_bc: # Behavior Cloning
        state = {k: v for k, v in context.items() if k in ['deter', 'stoch']}
        predicted_actions = policy(state)
        mse = jnp.mean(jnp.square(predicted_actions - context['action']))
        if self.config.only_bc:
          loss = self.config.loss_scales.bc * mse
        else:
          loss += self.config.loss_scales.bc * mse
        metrics.update({f'behavior_cloning_mse': mse})
      return loss, (traj, metrics)

    # ポリシーの更新
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)

    # 状態価値関数の更新
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics
  
  def combo_loss(self, traj_dict):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    # traj = {k: jnp.concatenate([img_traj[k], obs_traj[k]], axis=1) for k in img_traj.keys()}
    obs_traj, img_traj = traj_dict['obs_traj'], traj_dict['img_traj']
    traj = interp(obs_traj, img_traj, f=self.config.loss_scales.conservative_f)
    print(f'traj_shape: {next(iter((traj.values()))).shape}')
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss = loss.mean()

    # 正則化
    img_log_prob = self.actor(sg(img_traj)).log_prob(sg(img_traj['action']))[:-1]
    img_log_prob *= sg(img_traj['weight'])[:-1]
    if self.config.combo_random_policy:
      random_traj = traj_dict["random_traj"]
      ran_prob = jnp.ones(random_traj['action'][:-1].shape[:-1]) * 0.5
      ran_log_prob = jnp.log(ran_prob**random_traj['action'].shape[-1])
      ran_log_prob *= sg(random_traj['weight'])[:-1]
      print(f'img_log_prob.shape: {img_log_prob.shape}')
      print(f'ran_log_prob.shape: {ran_log_prob.shape}')
      img_ran_log_prob_diff = img_log_prob - ran_log_prob
      img_log_prob = jnp.stack([img_log_prob, ran_log_prob], axis=-1)
      img_log_prob = jax.nn.logsumexp(img_log_prob / self.config.logsumexp_temp, axis=-1)
    obs_log_prob = self.actor(sg(obs_traj)).log_prob(sg(obs_traj['action']))[:-1]
    obs_log_prob *= sg(obs_traj['weight'])[:-1]
    log_prob_term = img_log_prob - obs_log_prob
    log_prob_term *= self.config.loss_scales.conservative
    loss += log_prob_term.mean()

    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    metrics.update(jaxutils.tensorstats(img_log_prob, 'img_log_prob'))
    metrics.update(jaxutils.tensorstats(obs_log_prob, 'obs_log_prob'))
    if self.config.combo_random_policy:
      metrics.update(jaxutils.tensorstats(ran_log_prob, 'ran_log_prob'))
      metrics.update(jaxutils.tensorstats(img_ran_log_prob_diff, 'img_ran_log_prob_diff'))
    return loss, metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)
    if self.config.with_lagrange:
      self.lagrange_opt = jaxutils.Optimizer(name='lagrange_opt', **self.config.lagrange_opt)
      # with jax.transfer_guard("allow"):
      #   self.cql_log_alpha = jax.device_put(jnp.zeros(1), device=jax.devices("gpu")[0])
      self.cql_log_alpha = nj.Variable(jnp.zeros, (), name='cql_log_alpha')

  def train(self, traj, actor):
    if self.config.enable_combo:
      assert isinstance(traj, dict), f'the type of traj is invalid: {type(traj)}'
      target = {k: sg(self.score(v)[1]) for k, v in traj.items()}
      mets, metrics = self.opt(self.net, self.conservative_loss, traj, target, has_aux=True)
    else:
      target = sg(self.score(traj)[1])
      mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics
  
  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    metrics = jaxutils.tensorstats(dist.mean())
    if self.config.enable_rew_std_gen:
      dist_std = sg(dist.std())
      standardized_dist_std = (dist_std - dist_std.mean(axis=-1, keepdims=True)) / (dist_std.std(axis=-1, keepdims=True) + 1e-8)
      metrics.update(jaxutils.tensorstats(dist_std, 'dist_std'))
      metrics.update(jaxutils.tensorstats(standardized_dist_std, 'standardized_dist_std'))
      if self.config.rew_std_gen_sigmoid:
        target *= 1 / (1 + jnp.exp(standardized_dist_std)) # stdが大きいほど価値を小さく見積もる
      elif self.config.rew_std_gen_relu:
        target *= -jax.nn.relu(standardized_dist_std) + 1.0 # stdが大きいほど価値を小さく見積もる
      elif self.config.rew_std_gen_200:
        target -= 200 / (1 + jnp.exp(-standardized_dist_std)) # stdが大きいほど価値を小さく見積もる
      else:
        raise NotImplementedError()
    loss = -dist.log_prob(sg(target))
    
    if self.config.critic_slowreg == 'logprob':
        reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
        reg = -jnp.einsum(
            '...i,...i->...',
            sg(self.slow(traj).probs),
            jnp.log(dist.probs))
    else:
        raise NotImplementedError(self.config.critic_slowreg)
    
    loss += self.config.loss_scales.slowreg * reg

    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    return loss, metrics

  # def interp(self, a: dict, b: dict, n: int, f=0.5):
  #   '''
  #   sampling datapoints from a with probability f, or from b with probability 1-f.
  #   '''
  #   swap = lambda x: {k: v.transpose([1, 0] + list(range(2, len(v.shape)))) for k, v in x.items()}
  #   a, b = swap(a), swap(b)
  #   value_shape = next(iter((a.values()))).shape
  #   rng = jax.random.PRNGKey(0)
  #   choice_rng, indices_rng = jax.random.split(rng, 2)

  #   # f の確率で a から、1 - f の確率で b から選ぶ
  #   choices = jax.random.bernoulli(choice_rng, p=f, shape=(n,))

  #   # 選択したインデックスに基づいてサンプリング
  #   indices = jax.random.randint(indices_rng, shape=(n,), minval=0, maxval=value_shape[0])
  #   samples = {}
  #   for k in a.keys():
  #     choices_broadcasted = choices.reshape(n, *([1] * (len(a[k].shape) - 1)))
  #     samples[k] = jnp.where(choices_broadcasted, a[k][indices], b[k][indices])

  #   return swap(samples)
  
  def conservative_loss(self, traj, target):
    # V関数の損失
    metrics = {}
    traj = {key: {k: v[:-1] for k, v in trajectory.items()} for key, trajectory in traj.items()}

    img_traj = traj["img_traj"]
    obs_traj = traj["obs_traj"]
    if self.config.combo_random_policy:
      random_traj = traj["random_traj"]

    print(f'img_traj_shape: {list(traj["img_traj"].values())[0].shape}')
    print(f'obs_traj_shape: {list(traj["obs_traj"].values())[0].shape}')

    img_traj['target'] = target["img_traj"]
    obs_traj['target'] = target["obs_traj"]

    traj = interp(obs_traj, img_traj, f=self.config.loss_scales.conservative_f)
    target = traj['target']

    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    
    if self.config.critic_slowreg == 'logprob':
        reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
        reg = -jnp.einsum(
            '...i,...i->...',
            sg(self.slow(traj).probs),
            jnp.log(dist.probs))
    else:
        raise NotImplementedError(self.config.critic_slowreg)
    
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()

    # Conservative Policy Evaluationの計算
    expected_V_img = self.net(img_traj).mean() * sg(img_traj['weight'])
    expected_V_obs = self.net(obs_traj).mean() * sg(obs_traj['weight'])
    if self.config.combo_random_policy:
      expected_V_ran = self.net(random_traj).mean() * sg(random_traj['weight'])
      expected_V_img = jnp.stack([expected_V_img, expected_V_ran], axis=-1)
      expected_V_img = jax.nn.logsumexp(expected_V_img / self.config.logsumexp_temp, axis=-1)
    expected_V_img, expected_V_obs = expected_V_img.mean(), expected_V_obs.mean()
    conservative_loss = expected_V_img - expected_V_obs

    if self.config.with_lagrange:
      def lagrange_loss(conservative_loss):
        cql_alpha = jax.lax.clamp(0.0, jnp.exp(self.cql_log_alpha.read()), 1e6)
        conservative_loss = cql_alpha * (conservative_loss - self.config.lagrange_threshold)
        cql_alpha_loss = -conservative_loss
        return cql_alpha_loss, conservative_loss
      lagrange_metrics, conservative_loss = self.lagrange_opt(self.cql_log_alpha, lagrange_loss, conservative_loss, has_aux=True)

    conservative_loss *= self.config.loss_scales.conservative
    loss += conservative_loss

    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    metrics.update(jaxutils.tensorstats(expected_V_img, 'expected_V_img'))
    metrics.update(jaxutils.tensorstats(expected_V_obs, 'expected_V_obs'))
    metrics.update(jaxutils.tensorstats(conservative_loss, 'conservative_loss'))
    if self.config.combo_random_policy:
      metrics.update(jaxutils.tensorstats(expected_V_ran, 'expected_V_ran'))
    if self.config.with_lagrange:
      metrics.update(jaxutils.tensorstats(self.cql_log_alpha.read(), 'cql_log_alpha'))
      metrics.update(lagrange_metrics)
    return loss, metrics

  # def conservative_loss(self, img_traj, obs_traj, img_target, obs_target):
  #   # V関数の損失
  #   metrics = {}
  #   img_traj = {k: v[:-1] for k, v in img_traj.items()}
  #   obs_traj = {k: v[:-1] for k, v in obs_traj.items()}
  #   print(f'img_traj_shape: {list(img_traj.values())[0].shape}')
  #   print(f'obs_traj_shape: {list(obs_traj.values())[0].shape}')
  #   obs_traj['target'] = obs_target
  #   img_traj['target'] = img_target
  #   traj = {k: jnp.concatenate([img_traj[k], obs_traj[k]], axis=1) for k in img_traj.keys()}
  #   target = traj['target']

  #   dist = self.net(traj)
  #   loss = -dist.log_prob(sg(target))
    
  #   if self.config.critic_slowreg == 'logprob':
  #       reg = -dist.log_prob(sg(self.slow(traj).mean()))
  #   elif self.config.critic_slowreg == 'xent':
  #       reg = -jnp.einsum(
  #           '...i,...i->...',
  #           sg(self.slow(traj).probs),
  #           jnp.log(dist.probs))
  #   else:
  #       raise NotImplementedError(self.config.critic_slowreg)
    
  #   loss += self.config.loss_scales.slowreg * reg
  #   loss = (loss * sg(traj['weight'])).mean()

  #   # Conservative Policy Evaluationの計算
  #   expected_V_img, expected_V_obs = self.net(img_traj).mean(), self.net(obs_traj).mean()
  #   expected_V_img = (expected_V_img * sg(img_traj['weight'])).mean()
  #   expected_V_obs = (expected_V_obs * sg(obs_traj['weight'])).mean()
  #   conservative_loss = expected_V_img - expected_V_obs
  #   loss += self.config.loss_scales.conservative * conservative_loss

  #   loss *= self.config.loss_scales.critic
  #   metrics = jaxutils.tensorstats(dist.mean())
  #   metrics.update(jaxutils.tensorstats(expected_V_img, 'expected_V_img'))
  #   metrics.update(jaxutils.tensorstats(expected_V_obs, 'expected_V_obs'))
  #   metrics.update(jaxutils.tensorstats(conservative_loss, 'conservative_loss'))
  #   return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]
  
def interp(a: dict, b: dict, f=0.5):
  '''
  sampling datapoints from a with probability f, or from b with probability 1-f.
  '''

  swap = lambda x: {k: v.transpose([1, 0] + list(range(2, len(v.shape)))) for k, v in x.items()}
  a, b = swap(a), swap(b)
  value_shape = next(iter((a.values()))).shape
  n = value_shape[0]
  rng = jax.random.PRNGKey(0)

  # f の確率で a から、1 - f の確率で b から選ぶ
  choices = jax.random.bernoulli(rng, p=f, shape=(n,))

  # 選択したインデックスに基づいてサンプリング
  samples = {}
  for k in a.keys():
    choices_broadcasted = choices.reshape(n, *([1] * (len(a[k].shape) - 1)))
    samples[k] = jnp.where(choices_broadcasted, a[k], b[k])

  return swap(samples)