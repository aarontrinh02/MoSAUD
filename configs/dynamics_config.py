import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.model_cls = "DynamicsModel"
    config.hidden_dims = (256, 256)
    config.learning_rate = 3e-4
    config.weight_decay = 0.0
    config.dropout_rate = 0.1
    
    return config 