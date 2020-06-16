import torch.nn as nn
import torch
from bootstrap import Logger


class CountingRegression(nn.Module):
    def __init__(
        self,
        loss="mse",
        entropy_loss_weight=0.0,
    ):

        """
        entropy loss: term by term entropy
        """
        super().__init__()
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(loss)

        self.entropy_loss_weight = entropy_loss_weight
        Logger()(f"entropy_loss_weight={entropy_loss_weight}")

    def forward(self, net_out, batch):
        pred = net_out["pred"]
        out = dict()

        gt = (
            torch.tensor(batch["answer"])
            .to(device=pred.device)
            .reshape(pred.shape)
            .float()
        )

        loss = self.loss(net_out["pred"], gt)
        out["original_loss"] = loss

        if self.entropy_loss_weight != 0.0:
            entropy_weight = self.entropy_loss_weight
            out["entropy_weight"] = entropy_weight
            scores = net_out["scores"]  # prob for every item.
            entropy = -scores * torch.log2(scores) - (
                (1 - scores) * torch.log2(1 - scores)
            )
            entropy = entropy.mean(dim=1)  # b, 1: entropy for every batch item
            entropy = entropy.mean(dim=0).squeeze()  # (1)
            loss = loss + entropy_weight * entropy
            out["entropy_loss"] = entropy * entropy_weight
        out["loss"] = loss
        return out
