

class BaseDistrib(object):
    """
    The base class, from which the specific distributions are inherited
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, size):
        raise NotImplementedError
