def get_model(config):
    if config.network == "condensenc":
        from .condensenc import CondenseEncoderEpsNetwork
        return CondenseEncoderEpsNetwork(config)
    else:
        raise NotImplementedError("Unknown network: %s" % config.network)
