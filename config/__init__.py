import os
import yaml
import collections

def deep_update(d, u):
    for k, v in u.items():
        if not isinstance(d, collections.abc.Mapping):
            d = {}
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config() -> dict:
    env = os.getenv('ENVIRONMENT')
    mod = os.getenv('MODEL')

    if not env:
        raise ValueError('Missing environment: {}'.format(env))
    if env.endswith('-io'):
        env = env[:-3]
    if env not in ['development', 'development_production', 'alpha', 'beta', 'production', ]:
        raise ValueError('Invalid environment: {}'.format(env))
    # app.logger.info('STARTING: env={}'.format(env))
    stage = 'production' if env == 'production' else 'staging'

    if not mod:
        raise ValueError('Missing model: {}'.format(mod))
    if mod not in ['J-Engine', 'W-Engine']:
        raise ValueError('Invalid model: {}'.format(mod))
    # app.logger.info('STARTING: mod={}'.format(mod))

    with open('config/default.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    with open('config/{}.yaml'.format(env), 'r') as f:
        config_env = yaml.load(f, Loader=yaml.SafeLoader)

    if config_env:
        config = deep_update(config, config_env)

    config['env'] = env
    config['mod'] = mod
    config['stage'] = stage

    return config


config = load_config()
