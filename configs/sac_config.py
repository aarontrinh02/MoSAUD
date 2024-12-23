from ml_collections.config_dict import config_dict

from configs import td_config


def get_config():
    config = td_config.get_config()

    config.model_cls = "SACLearner"

    config.temp_lr = 3e-4
    config.discount = 0.99

    config.init_temperature = 0.05
    config.target_entropy = config_dict.placeholder(float)

    config.backup_entropy = True

    return config
