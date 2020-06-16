from bootstrap.compare3 import cli

if __name__ == "__main__":
    metrics = [
        # validation
        "logs:validation_epoch.tally_acc.overall:max",
        # "logs:validation_epoch.tally_acc.m-rel.overall:max",

        # simple
        "logs:eval_epoch.tally_acc.simple:max",
        # "logs:eval_epoch.tally_thresh_acc.simple:max",
        "logs:eval_epoch.tally_rmse.simple:max",

        # Compelex
        "logs:eval_epoch.tally_acc.complex:max",
        "logs:eval_epoch.tally_rmse.complex:max",
        "logs:eval_epoch.tally_acc.overall:max",
        "logs:nparams:max",


    ]
    cli(metrics, best="logs:validation_epoch.tally_acc.overall:max",
        sort="logs:eval_epoch.tally_acc.overall:max",
        bigtable=True)
