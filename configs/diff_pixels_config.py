from ml_collections.config_dict import config_dict

from configs import pixel_config


def get_config():
    config = pixel_config.get_config()
    del config.hidden_dims

    config.beta_schedule = "vp"
    config.T = 5
    config.use_layer_norm = True
    config.num_blocks = 3
    config.dropout_rate = 0.1
    config.tau = 0.001
    config.hidden_dim = 128
    config.lr = 3e-4
    config.decay_steps = 3e6

    return config
