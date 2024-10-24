from ml_collections.config_dict import config_dict

from configs import pixel_config


def get_config():
    config = pixel_config.get_config()

    config.model_cls = "DrQLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.discount = 0.99

    config.num_qs = 10
    config.num_min_qs = 2

    config.tau = 0.005
    config.init_temperature = 0.1
    config.critic_layer_norm = True
    config.backup_entropy = False
    config.target_entropy = config_dict.placeholder(float)

    config.weight_decay = 1e-3

    return config
