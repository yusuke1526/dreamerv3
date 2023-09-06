import numpy as np


class AutopilotAgent:
  def __init__(self, act_space, envs, **kwargs):
    self.act_space = act_space
    self._envs = [env.env._env for env in envs._envs]
    self._coef_acc = kwargs.get('coef_acc', 1.0)
    self._coef_steer = kwargs.get('coef_steer', 1.0)

  def policy(self, obs, state=None, mode="train"):
    act = {
        k: np.stack([self._get_action(env) for env in self._envs])
        for k, v in self.act_space.items()
        if k != "reset"
    }
    return act, state

  def _get_action(self, env):
    # Autopilotから制御信号を取得
    if env.ego:
      control = env.ego.get_control()
      throttle = control.throttle
      steer = control.steer
      brake = control.brake
    else:
      throttle = 0.0
      steer = 0.0
      brake = 0.0

    # Convert throttle and brake to acceleration
    if throttle > 0:
      acc = throttle
    else:
      acc = -brake

    acc = np.clip(acc, -1.0, 1.0) * self._coef_acc
    steer = np.clip(steer, -1.0, 1.0) * self._coef_steer

    # 制御信号をtf_agentsの形式に変換
    action = np.array([acc, -steer], dtype=np.float32)

    return action
