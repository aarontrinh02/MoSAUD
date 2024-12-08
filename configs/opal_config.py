import ml_collections

from configs import iql_config


def get_config():
    config = ml_collections.ConfigDict()
    config.discount = 0.99
    config.lr = 3e-4
    config.skill_dim = 8
    config.kl_coef = 0.1
    config.beta_coef = 0.25
    config.iql = iql_config.get_config()
    config.vae_hidden_dims = (256, 256)
    config.vae_encoder_hidden_size = 256
    config.latent_dim = 50
    return config
