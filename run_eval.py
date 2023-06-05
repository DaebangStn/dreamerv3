def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/dreamerv3/logdir/run2',
      'run.log_every': 30,  # Seconds
      'batch_size': 1,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      embodied.logger.WandBOutput(r'.*', logdir, config),
  ])

  from embodied.envs.crafter import Crafter
  from embodied.envs import from_gym
  env = Crafter(task='reward', outdir='~/dreamerv3/logdir/run2')  # Replace this with your Gym env.
  env = from_gym.FromGym(env._env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length).update({'from_checkpoint': '/home/geon/dreamerv3/logdir/run1/checkpoint.ckpt'})

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  embodied.run.eval_only(agent, env, logger, args)

if __name__ == '__main__':
  main()
