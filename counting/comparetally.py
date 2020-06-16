import argparse
from bootstrap.compare3 import cli

if __name__ == "__main__":
    metrics = [
        "logs:eval_epoch.tally_acc.simple:max",
        "logs:eval_epoch.tally_rmse.simple:min",
        "logs:eval_epoch.tally_acc.complex:max",
        "logs:eval_epoch.tally_rmse.complex:min",
        # "logs:eval_epoch.loss:min",
        "logs:eval_epoch.tally_acc.overall:max",

    ]
    cli(metrics, best="logs:eval_epoch.tally_acc.overall:max", bigtable=True)
