import importlib

from ..lib.options import Options
from ..lib.logger import Logger
from bootstrap.datasets.dataset import BootstrapDataset
from importlib import import_module
import traceback


def wrap(dataset, opt, shuffle=True):
    if not hasattr(dataset, "make_batch_loader"):
        dataset = BootstrapDataset(
            dataset,
            opt["batch_size"],
            nb_threads=opt["nb_threads"],
            shuffle=shuffle,
            pin_memory=Options()["misc.cuda"],
        )
        return dataset
    else:
        return dataset


def factory(engine=None):
    Logger()("Creating dataset...")
    opt = Options()["dataset"]
    if "import" in Options()["dataset"]:
        # import looks like "yourmodule.datasets.factory"
        module = importlib.import_module(Options()["dataset"]["import"])
        dataset = module.factory(engine=engine)

    else:
        module, class_name = opt["name"].rsplit(".", 1)
        try:
            cls = getattr(import_module(module), class_name)
        except ImportError:
            Logger()(f"Unable to import class {module}:{class_name}")
            traceback.print_exc()

        dataset = {}
        if opt["train_split"] is not None:
            d = cls(
                **opt["params"],
                split=opt["train_split"],
                shuffle=True,
                batch_size=opt["batch_size"],
            )
            Logger()(f"Train set created, of type {type(d)}")
            dataset["train"] = wrap(d, opt, shuffle=True)
            Logger()(
                "Training will take place on {}set ({} items)".format(
                    dataset["train"].split, len(dataset["train"])
                )
            )

        if opt["eval_split"] is not None:
            d = cls(
                **opt["params"],
                split=opt["eval_split"],
                shuffle=False,
                batch_size=opt["batch_size"],
            )
            Logger()(f"Eval set created, of type {type(d)}")
            dataset["eval"] = wrap(d, opt, shuffle=False)
            Logger()(
                "Evaluation will take place on {}set ({} items)".format(
                    dataset["eval"].split, len(dataset["eval"])
                )
            )

        if opt.get("validation_split", None) is not None:
            d = cls(
                **opt["params"],
                split=opt["validation_split"],
                shuffle=False,
                batch_size=opt["batch_size"],
            )
            Logger()(f"Eval set created, of type {type(d)}")
            dataset["validation"] = wrap(d, opt, shuffle=False)
            Logger()(
                "Validation will take place on {}set ({} items)".format(
                    dataset["validation"].split, len(dataset["validation"])
                )
            )

    return dataset
