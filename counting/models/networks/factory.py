import sys
import torch
import traceback
from importlib import import_module

from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from bootstrap.lib.logger import Logger


def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()["model.network"]

    if True:
        module, class_name = opt["name"].rsplit(".", 1)
        try:
            cls = getattr(
                import_module("." + module, "counting.models.networks"), class_name
            )
        except:
            traceback.print_exc()
            Logger()(f"Error importing class {module}, {class_name}")
            sys.exit(1)
        print("Network parameters", opt["parameters"])
        # check if @ in parameters
        print("checking if @ in parameters")
        parameters = opt.get("parameters", {}) or {}
        for key, value in parameters.items():  # TODO intégrer ça à bootstrap
            if value == "@dataset":
                print("loading dataset")
            elif value == "@engine":
                opt["parameters"][key] = engine
            elif value == "@aid_to_ans":
                opt["parameters"][key] = dataset.aid_to_ans
            elif value == "@ans_to_aid":
                opt["parameters"][key] = dataset.ans_to_aid
        net = cls(
            **parameters,
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
        )

    if Options()["misc.cuda"] and torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net
