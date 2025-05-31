import os
import yaml

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']

    Shamelessly stolen from: https://stackoverflow.com/a/13520518
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def load_config(path: str) -> DotDict:
    with open(path, "r") as f:
        dct = yaml.safe_load(f)
        return DotDict(dct)


def load_env(path: str) -> None:
    """ Function for loading .env file for secret API keys etc. and write them to the global enviroment """
    with open(path, 'r') as fh:
        vars_dict = dict(
            tuple(line.replace('\n', '').split('=')) for line in fh.readlines() if not line.startswith('#')
        )
        os.environ.update(vars_dict)
