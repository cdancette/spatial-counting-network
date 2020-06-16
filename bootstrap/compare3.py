import json
import numpy as np
import argparse
from os import path as osp
import os
from tabulate import tabulate
import joblib
import yaml
from collections import defaultdict, namedtuple


ResultPoint = namedtuple("ResultPoint", ["epoch", "value", "params"], defaults=(None,))

empty_result = ResultPoint(epoch=None, value=None)


def load_json(logs):
    with open(logs) as f:
        data = json.load(f)
    return data


def print_results(
    results_dict,
    tablefmt=None,
    format=".2f",
    big_table=False,
    best=None,
    sort=None,
    transpose=False,
):
    """
    results_dict[metric_full_name] = {"order": order, "res": {'dir_exp': ResultPoint}}
    """
    if sort is not None:
        best = sort

    all_params = set()

    for name in results_dict:
        order = results_dict[name]["order"]
        results = results_dict[name]["res"]
        res_list = []
        # sort results

        for dirname in results:
            if type(results[dirname]) == list:
                breakpoint()

        sorted_dirs = sorted(
            results,
            key=lambda dirname: results[dirname].value,
            reverse=(order == "max"),
        )

        for dir_name in sorted_dirs:
            epoch, res, params = results[dir_name]
            res = f"{res:.2f}"
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
        exp_dirs = list(exp_dirs)

        if best is not None:
            # none values
            exp_dirs_none = [e for e in exp_dirs if results_dict[best]["res"] is None]
            exp_dirs_not_none = [
                e for e in exp_dirs if results_dict[best]["res"] is not None
            ]
            exp_dirs = sorted(
                exp_dirs_not_none,
                key=lambda e: results_dict[best]["res"].get(e, empty_result).value
                or False,
                reverse=(results_dict[best]["order"] == "max"),
            )
            exp_dirs = exp_dirs + exp_dirs_none
        var = []
        common_prefix = ""  # os.path.commonprefix(exp_dirs)
        exp_dirs_print = [e[len(common_prefix) :] for e in exp_dirs]
        print("headers", headers)
        print(f"## Common prefix: {common_prefix}")
        for i, exp_dir in enumerate(exp_dirs):
            row = []
            row.append(exp_dirs_print[i])
            results_epochs = [
                results_dict[metric]["res"].get(exp_dir, None) for metric in metrics
            ]
            results = [
                f"{r.value:.2f}" if r is not None else "" for r in results_epochs
            ]
            epochs = [r.epoch for r in results_epochs if r is not None]
            # assert min(epochs) == max(
            #     epochs
            # ), "Warning !! Not all values have same epoch. add --best or disable --bigtable"
            epoch = epochs[0]
            row.extend(results)
            row.append(str(epoch))
            var.append(row)

        # if tablefmt is None:
        #     from rich.table import Table
        #     from rich.console import Console
        #     console = Console()
        #     table = Table(show_header=True, header_style="bold blue")
        #     for header in headers:
        #         table.add_column(header.lstrip('logs:').rstrip(':max').rstrip(':min'))
        #     for row in var:
        #         table.add_row(*row)
        #     console.print(table)
        # else:
        if tablefmt == "parsable":
            table = [headers] + var
            if transpose:
                table = list(map(list, zip(*table)))
            for row in table:
                print(";".join(row))
        else:
            headers = [h.replace(".", ".\n").lstrip("logs:") for h in headers]
            print(tabulate(var, tablefmt=tablefmt or "simple", headers=headers))

        # table = Texttable(max_width=110)
        # table.set_deco(Texttable.HEADER)
        # # table.header(headers)
        # # print(len(headers))
        # # print(len(var[0]))
        # table.add_row(headers)
        # table.add_rows(var)
        # print(table.draw())


def get_epochs(logs, metric_name):
    if "eval_epoch" in metric_name:
        if "eval_epoch.epoch" not in logs:
            return None
        return logs["eval_epoch.epoch"]
    elif "train_epoch" in metric_name:
        return logs["train_epoch.epoch"]
    elif "validation_epoch" in metric_name:
        return logs["validation_epoch.epoch"]
    else:
        return None
    # else:
    #     raise ValueError(metric_name)


def filter_epoch(array, min_epoch=None, max_epoch=None):
    """
    array: [(epoch, value), (epoch, value), ...]
    Returns: the same array, with only epochs between min_epoch and max_epoch
    """
    if min_epoch is None:
        min_epoch = float("-inf")
    if max_epoch == -1 or max_epoch is None:
        max_epoch = float("inf")
    return [
        point
        for point in array
        if point.epoch >= min_epoch and point.epoch <= max_epoch
    ]


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


# default params to None


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

    if best and best not in metrics:
        metrics.append(best)

    for metric_name in metrics:
        if type(metric_name) == str:
            if "+" in metric_name:
                for m in metric_name.split("+"):
                    log_name, metric, order = m.split(":")
                    m = (log_name, metric, order, metric_name)
                    if m not in metrics_list:
                        metrics_list.append(m)
            else:
                log_name, metric, order = metric_name.split(":")
                m = (log_name, metric, order, metric_name)
                if m not in metrics_list:
                    metrics_list.append((log_name, metric, order, metric_name))
                if args.add_logs_name:
                    metrics_list.append(
                        (args.add_logs_name, metric, order, metric_name)
                    )
        elif type(metric_name) == list:
            log_name, metric, order = metric_name
            m = (log_name, metric, order, metric_name)
            if m not in metrics_list:
                metrics_list.append((log_name, metric, order, metric_name))

    log_names = {log_name for log_name, _, _, _ in metrics_list}

    # loading json results
    # dir_exp -> log_name -> logs (= metric_name -> values)
    print("loading results")
    logs_for_dir = dict()
    # results = Parallel(n_jobs=30)(delayed(open_results)(dir_exp, log_names) for dir_exp in dir_exps)
    logs = [open_results(dir_exp, log_names) for dir_exp in dir_exps]
    for i, dir_e in enumerate(dir_exps):
        logs_for_dir[dir_e] = logs[i]

    # load options for params
    params = defaultdict(dict)  # dir_exp -> param_name -> param_value
    if args.param:
        for dir_e in dir_exps:
            options = load_option(osp.join(dir_e, "options.yaml"))
            for p in args.param:
                params[dir_e][p] = get_param(options, p)

    # now process this to make analysis easier.
    # metrics_for_dir: dir_exp -> metric_name -> ResultPoint
    metrics_for_dir = dict()
    for dir_e in dir_exps:
        metrics_for_dir[dir_e] = dict()
        for log_name, metric, order, metric_fullname in metrics_list:
            if (
                log_name in logs_for_dir[dir_e]
                and metric in logs_for_dir[dir_e][log_name]
            ):
                epochs = get_epochs(logs_for_dir[dir_e][log_name], metric_fullname)
                if epochs is None:
                    # create epochs
                    epochs = list(range(len(logs_for_dir[dir_e][log_name][metric])))
                metrics_for_dir[dir_e][metric_fullname] = [
                    ResultPoint(epoch=e, value=v, params=params[dir_e])
                    for (e, v) in zip(epochs, logs_for_dir[dir_e][log_name][metric])
                ]
            orders_dict[metric_fullname] = order

    # breakpoint()
    # manage sums
    for metric_name in metrics:
        if "+" in metric_name:
            m1, m2 = metric_name.split("+")
            for dir_e in dir_exps:
                if m1 in metrics_for_dir[dir_e] and m2 in metrics_for_dir[dir_e]:
                    assert all(
                        p1.epoch == p2.epoch
                        for (p1, p2) in zip(
                            metrics_for_dir[dir_e][m1], metrics_for_dir[dir_e][m2]
                        )
                    )
                    metrics_for_dir[dir_e][metric_name] = [
                        ResultPoint(
                            epoch=p1.epoch,
                            value=(p1.value + p2.value) / 2,
                            params=p1.params,
                        )
                        for p1, p2 in zip(
                            metrics_for_dir[dir_e][m1], metrics_for_dir[dir_e][m2]
                        )
                    ]
            orders_dict[metric_name] = orders_dict[m1]

    # find best epoch for each dir_exp
    if best:
        best_epochs = dict()  # dir_exp -> best_epoch
        b_order = orders_dict[best]
        for dir_e in dir_exps:
            if best not in metrics_for_dir[dir_e]:
                print(f"Best Metric {best} not in {dir_e}.")
                continue
            res = metrics_for_dir[dir_e][best]
            res = filter_epoch(res, min_epoch, max_epoch)
            if b_order == "max":
                f = max
            elif b_order == "min":
                f = min
            res = f(res, key=lambda x: x.value)
            best_epochs[dir_e] = res.epoch
    # computing results

    print("computing results")
    results = {}  # metrics -> dir_logs -> results
    for log_name, metric, order, fullname in metrics_list:
        if type(fullname) == list:
            print("what is happening? fullname is list..")
            fullname = ":".join(fullname)
        results[fullname] = {"res": {}, "order": order}
        for dir_e in dir_exps:
            if fullname in metrics_for_dir[dir_e]:
                res = metrics_for_dir[dir_e][fullname]
                res = filter_epoch(res, min_epoch, max_epoch)
                if len(res) == 1:
                    results[fullname]["res"][dir_e] = res[0]
                else:
                    if res:
                        if last:
                            res = res[-1]  # pick last result
                        elif best:
                            if dir_e in best_epochs:
                                # breakpoint()
                                try:
                                    best_epoch = best_epochs[dir_e]
                                    res = [r for r in res if r.epoch == best_epoch][0]
                                except IndexError:
                                    res = None
                            else:
                                res = None  # res[-1]
                        else:
                            if order == "max":
                                f = max
                            elif order == "min":
                                f = min
                            res = f(res, key=lambda x: x.value)
                        if res is not None:
                            results[fullname]["res"][dir_e] = res
    print_results(
        results,
        tablefmt=args.tablefmt,
        format=args.format,
        big_table=args.bigtable,
        best=args.best,
        sort=args.sort,
        transpose=args.transpose,
    )


def cli(
    metrics=[
        "logs:eval_epoch.accuracy_top1:max",
        "logs:eval_epoch.accuracy_top5:max",
        "logs:eval_epoch.loss:min",
    ],
    best=None,
    sort=None,
    bigtable=False,
    last=False,
):
    """
    Call this function in your module to override default metrics and best.
    For example, create a file compare_f1.py containing
        cli(metrics=['logs:eval_epoch.accuracy', logs:eval_epoch.f1'])
    
    This will generate a command line.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--min-epoch", default=float("-inf"), type=int)
    parser.add_argument("-n", "--nb_epochs", default=-1, type=int)
    parser.add_argument("-d", "--dir_logs", default="", type=str, nargs="*")

    parser.add_argument(
        "-m", "--metrics", type=str, nargs="+", default=metrics,
    )
    parser.add_argument("--add-logs-name",)

    parser.add_argument(
        "-b",
        "--best",
        type=str,
        nargs="?",
        default=best,
        const=None,
        help="If this is specified, for each metric, the epoch specified will "
        "be the one that is best for this metric",
    )
    parser.add_argument("--sort", default=sort)
    parser.add_argument("--param", nargs="*", help="additional parameters to display")
    parser.add_argument(
        "--sort-param",
        help="sort by a specific parameter. param must be specified with --param",
    )
    parser.add_argument(
        "--last", action="store_true", help="select only last metric", default=last
    )
    parser.add_argument("--tablefmt")
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--format", default=".2f")
    parser.add_argument("--bigtable", action="store_true", default=bigtable)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
