import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.discount = 0.99
    config.temperature = 1.0
    config.expectile = 0.9
    config.lr = 3e-4
    config.tau = 5e-3
    config.num_qs = 2
    config.hidden_dims = (256, 256)
    config.critic_layer_norm = False
    config.faster_actor_update = False
    config.policy_extraction = "awr"
    return config
