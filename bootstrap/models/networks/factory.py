import importlib

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

def factory(engine=None):

    Logger()('Creating network...')

    if 'import' in Options()['model']['network']:
        module = importlib.import_module(Options()['model']['network']['import'])
        network = module.factory(engine)

    else:
        module, class_name = opt['name'].rsplit('.', 1)
        cls = getattr(import_module('.' + module, 'counting.models.networks'), class_name)
        print("Network parameters", opt['parameters'])
        # check if @ in parameters
        print("checking if @ in parameters")
        for key, value in opt['parameters'].items():  # TODO intégrer ça à bootstrap
            if value.startswith("@"):
                try:
                    output  = eval(value[1:])
                    opt['parameters'][key] = output
                except:
                    pass
        net = cls(
            **opt['parameters'],
        )

        raise ValueError()

    Logger()(f'Network created, of type {type(network)}...')

    return network
