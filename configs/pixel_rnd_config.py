from configs import pixel_config


def get_config():
    config = pixel_config.get_config()

    config.model_cls = "PixelRND"
    config.lr = 3e-4
    config.hidden_dims = (256, 256)
    config.coeff = 8.0
    return config
