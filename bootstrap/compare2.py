import json
import numpy as np
import argparse
from os import path as osp
from tabulate import tabulate
import joblib
import yaml
from collections import defaultdict

try:
    from texttable import Texttable
except:
    pass


def load_json(logs):
    with open(logs) as f:
        data = json.load(f)
    return data


def print_results(
    results_dict, tablefmt=None, sort_param=None, format=".2f", big_table=False
):
    """
    results_dict[metric_full_name] = {"order": order, "res": {'dir_exp': (epoch, value)}}
    """
    all_params = set()
    for name in results_dict:
        order = results_dict[name]["order"]
        results = results_dict[name]["res"]
        res_list = []
        # sort results
        sorted_dirs = sorted(results, key=lambda dirname: results[dirname].value, reverse=(order == "max"))
        for dir_name in sorted_dirs:
            epoch, res, params = results[dir_name]
            res = f'{res:.2f}'
            r_list = [dir_name, res, epoch]
            for p in params:
                all_params.add(p)
                r_list.append(params[p])
            res_list.append(r_list)
        
        if not res_list:
            continue
        print()
        print("## " + name)
        headers = ["Dir", "Score", "Epoch"]
        for p in all_params:
            headers.append(p.split(".")[-1])
        if tablefmt == "parsable":
            print(";".join(headers))
            for r in res_list:
                print(";".join(map(str, r)))
        else:
            print(tabulate(res_list, headers=headers, tablefmt=tablefmt or "github"))

    if big_table:
        headers = (
            ["dir"]
            + [":".join(name.split(":")[:-1]) for name in results_dict]
            + ["epoch"]
        )
        metrics = list(results_dict.keys())

        # exp_dir -> metric_name -> value
        exp_dirs = set()
        for metric in results_dict:
            exp_dirs = exp_dirs.union(results_dict[metric]["res"].keys())

        var = []
        for exp_dir in exp_dirs:
            row = []
            row.append(exp_dir)
            results_epochs = [
                results_dict[metric]["res"].get(exp_dir, None) for metric in metrics
            ]
            results = [
                f"{r.value:.2f}" if r is not None else "" for r in results_epochs
            ]
            epochs = [r.epoch for r in results_epochs if r is not None]
            assert min(epochs) == max(
                epochs
            ), "Warning !! Not all values have same epoch. add --best or disable --bigtable"
            epoch = epochs[0]
            row.extend(results)
            row.append(epoch)
            var.append(row)

        print("headers", headers)
        # if tablefmt == "latex":
        # var = [headers] + var

        print(tabulate(var, tablefmt=tablefmt))

        # table = Texttable(max_width=110)
        # table.set_deco(Texttable.HEADER)
        # # table.header(headers)
        # # print(len(headers))
        # # print(len(var[0]))
        # table.add_row(headers)
        # table.add_rows(var)
        # print(table.draw())


def get_epoch(logs, metric_name):
    if metric_name.startswith("eval_epoch"):
        if "eval_epoch.epoch" not in logs:
            return None
        return logs["eval_epoch.epoch"]
    elif metric_name.startswith("train_epoch"):
        return logs["train_epoch.epoch"]
    else:
        raise ValueError(metric_name)


def filter_epoch(array, min_epoch=None, max_epoch=None):
    """
    array: [(epoch, value), (epoch, value), ...]
    Returns: the same array, with only epochs between min_epoch and max_epoch
    """
    if min_epoch is None:
        min_epoch = float("-inf")
    if max_epoch == -1 or max_epoch is None:
        max_epoch = float("inf")
    return [point  for point in array if point.epoch >= min_epoch and point.epoch <= max_epoch]


def load_option(option_path):
    with open(option_path) as f:
        options = yaml.load(f)
    return options


def get_param(options, param_name):
    param = param_name.split(".")
    for p in param:
        options = options[p]
    return options


def open_results(dir_exp, log_names):
    res = dict()
    for log_name in log_names:
        path = osp.join(dir_exp, f"{log_name}.json")
        if osp.exists(path):
            try:
                res[log_name] = load_json(path)
            except json.decoder.JSONDecodeError:
                continue
    return res


from collections import namedtuple
# default params to None
ResultPoint = namedtuple('ResultPoint', ['epoch', 'value', 'params'], defaults=(None,))


def main(args):
    max_epoch = args.nb_epochs
    min_epoch = args.min_epoch if hasattr(args, "min_epoch") else None
    dir_exps = args.dir_logs
    metrics = args.metrics
    last = args.last
    best = args.best
    if last:
        best = None
        args.best = None
    # assert not (best and last)

    # metrics = defaultdict(list) # json_file -> metric
    # Get metric list: [(log_name, metric, order, full_metric_name)]
    # metric_dict: logs_name -> metric_name -> order
    metrics_list = []
    orders_dict = dict()
    
    for metric_name in metrics:
        if type(metric_name) == str:
            if "+" in metric_name:
                for m in metric_name.split("+"):
                    log_name, metric, order = m.split(":")
                    metrics_list.append((log_name, metric, order, metric_name))
            else:
                log_name, metric, order = metric_name.split(":")
                metrics_list.append((log_name, metric, order, metric_name))
        elif type(metric_name) == list:
            log_name, metric, order = metric_name
            metrics_list.append((log_name, metric, order, metric_name))
        orders_dict[(log_name, metric)] = order

    log_names = {log_name for log_name, _, _, _ in metrics_list}

    # loading json results
    # dir_exp -> log_name -> logs (= metric_name -> values)
    logs_for_dir = defaultdict(dict)
    print("loading results")
    # results = Parallel(n_jobs=30)(delayed(open_results)(dir_exp, log_names) for dir_exp in dir_exps)
    logs = [open_results(dir_exp, log_names) for dir_exp in dir_exps]
    for i, dir_e in enumerate(dir_exps):
        logs_for_dir[dir_e] = logs[i]
    print("done")

    # load options for params
    params = defaultdict(dict)  # dir_exp -> param_name -> param_value
    if args.param:
        for dir_e in dir_exps:
            options = load_option(osp.join(dir_e, "options.yaml"))
            for p in args.param:
                params[dir_e][p] = get_param(options, p)

    # find best epoch for each dir_exp
    if best:
        b_log_name, b_metric = best.split(":")
        b_order = orders_dict[(b_log_name, b_metric)]
        best_epochs = dict()  # dir_exp -> best_epoch
        for dir_e in dir_exps:
            logs = logs_for_dir[dir_e]
            if b_log_name not in logs:
                print(f"{b_log_name} not in logs. for {dir_e}")
                continue
            epochs = get_epoch(logs[b_log_name], b_metric)
            if epochs is None:
                continue
            if b_metric not in logs[b_log_name]:
                # TODO: for now return last epoch
                print(f"Warning: {b_metric} not in {b_log_name}")
                best_epochs[dir_e] = epochs[-1]
                continue
            res = [ResultPoint(epoch=e, value=v) for (e, v) in zip(epochs, logs[b_log_name][b_metric])]
            # res = list(zip(epochs, logs[b_log_name][b_metric]))
            res = filter_epoch(res, min_epoch, max_epoch)
            if b_order == "max":
                f = max
            elif b_order == "min":
                f = min
            res = f(res, key=lambda x: x[1])
            best_epochs[dir_e] = res[0]

    # computing results
    results = {}  # metrics -> dir_logs -> results
    for log_name, metric, order, fullname in metrics_list:
        if type(fullname) == list:
            fullname = ":".join(fullname)
        results[fullname] = {"res": {}, "order": order}
        for dir_e in dir_exps:
            logs = logs_for_dir[dir_e]
            if log_name in logs and metric in logs[log_name]:
                epochs = get_epoch(logs[log_name], metric)
                p = params[dir_e]
                # res = [(epoch, score, param), .. ]
                res = [ResultPoint(epoch=e, value=v, params=p) for (e, v) in zip(epochs, logs[log_name][metric])]
                # res = 
                # res = list(zip(epochs, logs[log_name][metric]))
                res = filter_epoch(res, min_epoch, max_epoch)
                if res:
                    if last:
                        res = res[-1]  # pick last result
                    elif best:
                        res = [r for r in res if r.epoch == best_epochs[dir_e]][0]
                    else:
                        if order == "max":
                            f = max
                        elif order == "min":
                            f = min
                        res = f(res, key=lambda x: x.value)
                    results[fullname]["res"][dir_e] = res
    print_results(
        results, tablefmt=args.tablefmt, format=args.format, big_table=args.bigtable
    )


def cli(
    metrics=[
        "logs:eval_epoch.accuracy_top1:max",
        "logs:eval_epoch.accuracy_top5:max",
        "logs:eval_epoch.loss:min",
    ],
    best=None,
):
    """
    Call this function in your module to override default metrics and best.
    For example, create a file compare_f1.py containing
        cli(metrics=['logs:eval_epoch.accuracy', logs:eval_epoch.f1'])
    
    This will generate a command line.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--min-epoch", default=0, type=int)
    parser.add_argument("-n", "--nb_epochs", default=-1, type=int)
    parser.add_argument("-d", "--dir_logs", default="", type=str, nargs="*")

    parser.add_argument(
        "-m", "--metrics", type=str, action="append", default=metrics,
    )
    parser.add_argument(
        "-b",
        "--best",
        type=str,
        default=best,
        help="If this is specified, for each metric, the epoch specified will "
        "be the one that is best for this metric",
    )
    parser.add_argument("--parsable", action="store_true", help="parsable")
    parser.add_argument("--param", nargs="*", help="additional parameters to display")
    parser.add_argument(
        "--sort",
        help="sort by a specific parameter. param must be specified with --param",
    )
    parser.add_argument("--last", action="store_true", help="select only last metric")
    parser.add_argument("--tablefmt")
    parser.add_argument("--format", default=".2f")
    parser.add_argument("--bigtable", action="store_true")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
