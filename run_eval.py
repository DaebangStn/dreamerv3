import embodied

def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/logdir/run2',
      'run.log_every': 30,  # Seconds
      'batch_size': 1,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
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

  import crafter
  from embodied.envs import from_gym
  env = crafter.Env()  # Replace this with your Gym env.
  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)

  checkpoint = embodied.Checkpoint('~/logdir/run1/checkpoint.ckpt')
  checkpoint.load_or_save()
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)

  agent = checkpoint.agent
  embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
