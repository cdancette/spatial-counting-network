from bootstrap.lib.options import Options
from block.models.criterions.vqa_cross_entropy import VQACrossEntropyLoss
from .counting_regression import CountingRegression


def factory(engine, mode):
    name = Options()["model.criterion.name"]
    opt = Options()["model.criterion"]
    # if split == "test" and "tdiuc" not in Options()["dataset.name"]:
    #     return None
    if name == "vqa_cross_entropy":
        criterion = VQACrossEntropyLoss()
    elif name == "counting-regression":
        criterion = CountingRegression(
            loss=opt.get("loss", "mse"),
            entropy_loss_weight=opt.get("entropy_loss_weight", 0.0),
        )
    else:
        raise ValueError(name)
    return criterion
