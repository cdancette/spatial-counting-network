import torch.nn as nn
from bootstrap.lib.options import Options
from block.optimizers.lr_scheduler import ReduceLROnPlateau
from block.optimizers.lr_scheduler import BanOptimizer


def factory(model, engine):
    opt = Options()["optimizer"]

    optimizer = BanOptimizer(
        engine,
        name=Options()["optimizer"].get("name", "Adamax"),
        lr=Options()["optimizer"]["lr"],
        gradual_warmup_steps=Options()["optimizer"].get(
            "gradual_warmup_steps", [0.5, 2.0, 4]
        ),
        lr_decay_epochs=Options()["optimizer"].get("lr_decay_epochs", [10, 20, 2]),
        lr_decay_rate=Options()["optimizer"].get("lr_decay_rate", 0.25),
    )

    if Options()["model.network.name"] == "rcn.RCN":

        def clip_gradients():
            nn.utils.clip_grad_norm_(
                engine.model.network.parameters(), opt["clip_norm"]
            )

        # add hook
        engine.register_hook("train_on_backward", clip_gradients)

    if opt.get("lr_scheduler", None):
        optimizer = ReduceLROnPlateau(optimizer, engine, **opt["lr_scheduler"])

    if opt.get("init", None) == "glorot":
        for p in model.network.parameters():
            if p.dim() == 1:
                p.data.fill_(0)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                raise ValueError(p.dim())

    return optimizer
