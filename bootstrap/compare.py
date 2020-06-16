import json
import numpy as np
import argparse
from os import path as osp
from tabulate import tabulate
import traceback    

def load_values(dir_logs, metrics, nb_epochs=-1, best=None, last=False):
    json_files = {}
    values = {}

    # load argsup of best
    if best:
        if best['json'] not in json_files:
            with open(osp.join(dir_logs, f'{best["json"]}.json')) as f:
                json_files[best['json']] = json.load(f)

        jfile = json_files[best['json']]
        try:
            vals = jfile[best['name']]
        except:
            print(dir_logs)
            traceback.print_exc()
            #import ipdb;ipdb.set_trace()
        end = len(vals) if nb_epochs == -1 else nb_epochs
        argsup = np.__dict__[f'arg{best["order"]}'](vals[:end])

    # load logs
    for mkey, metric in metrics.items():
        # open json_files
        if metric['json'] not in json_files:
            path_json = osp.join(dir_logs, f'{metric["json"]}.json')
            if not osp.isfile(path_json):
                print(f'Warning: not found {path_json}')
                continue
            with open(path_json) as f:
                json_files[metric['json']] = json.load(f)

        jfile = json_files[metric['json']]
        #import ipdb; ipdb.set_trace()

        if 'train' in metric['name']:
            epoch_key = 'train_epoch.epoch'
        else:
            epoch_key = 'eval_epoch.epoch'

        if epoch_key in jfile:
            epochs = jfile[epoch_key]
        elif 'epoch' in jfile:
            epochs = jfile['epoch']
        
        try:
            vals = jfile[metric['name']]
        except:
            values[metric['id']] = -1, -1
            continue
        if not best:
            end = len(vals) if nb_epochs == -1 else nb_epochs
            argsup = np.__dict__[f'arg{metric["order"]}'](vals[:end])
        if last:
            argsup = -1
        try:
            values[metric['id']] = epochs[argsup], vals[argsup]
        except:
            try:
                values[metric['id']] = epochs[argsup-1], vals[argsup-1]
            except:
                print('Warning: {} for {}'.format(metric['name'], dir_logs))
                print('         epochs[argsup-1] not possible')
                print('         argsup-1: {}'.format(argsup-1))
                print('         epochs: {}'.format(epochs))
    return values

def main(args):
    if not hasattr(args, 'last'):
        args.last = False
    dir_logs = {}
    for raw in args.dir_logs:
        tmp = raw.split(':')
        if len(tmp) == 2:
            key, path = tmp
        elif len(tmp) == 1:
            path = tmp[0]
            key = osp.basename(osp.normpath(path))
        else:
            raise ValueError(raw)
        dir_logs[key] = path

    metrics = {}
    for json, name, order in args.metrics:
        metrics[f'{json}_{name}'] = {
            'id': json + '_' + name,
            'json': json,
            'name': name,
            'order': order,
        }

    if args.best:
        json, name, order = args.best
        best = {
            'id': json + '_' + name,
            'json': json,
            'name': name,
            'order': order
        }
    else:
        best = None

    logs = {}
    for name, dir_log in dir_logs.items():
        if osp.isfile(osp.join(dir_log, 'logs.json')):
            try:
                logs[name] = load_values(dir_log, metrics,
                    nb_epochs=args.nb_epochs,
                    best=best, 
                    last=args.last)
            except:
                continue
        else:
            print('Warning: logs.json not found in {}'.format(dir_log))
    
    for mkey, metric in metrics.items():
        names = []
        values = []
        epochs = []
        for name, vals in logs.items():
            if metric['id'] in vals:
                names.append(name)
                epoch, value = vals[metric['id']]
                epochs.append(epoch)
                values.append(value)
        if values:
            values_names = sorted(zip(values, names, epochs), reverse=metric['order']=='max')
            values_names = [[i + 1, name, value, epoch] for i, (value, name, epoch) in enumerate(values_names)]
            print('\n\n## {} : {}\n'.format(metric['json'], metric['name']))
            print(tabulate(values_names, headers=['Place', 'Method', 'Score', 'Epoch']))
            print("mean", np.mean(values))
            print("std", np.std(values))

    def sort_func(x):
        try:
            val = x[4][:5]
            val = float(val)
        except:
            val = 0
        return val

    # big matrix
    print('\n')
    print('## Big table\n')
    table = []
    for name, vals in logs.items():
        line = [name]
        for mkey, metric in metrics.items():
            if metric['id'] in vals:
                epoch, value = vals[metric['id']]
                value = format(value, '.2f')
            else:
                epoch, value = None, None
            line.append(f"{value} ({epoch})")
        table.append(line)
    # sort table
    table = sorted(table, key=sort_func, reverse=True)
    #print(tabulate(table, headers=['Method', *metrics.keys()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-m', '--metrics', type=str, action='append', nargs=3,
                        metavar=('json', 'name', 'order'),
                        default=[['logs', 'eval_epoch.accuracy_top1', 'max'],
                                 ['logs', 'eval_epoch.accuracy_top5', 'max'],
                                 ['logs', 'eval_epoch.loss', 'min']])
    parser.add_argument('-b', '--best', type=str, nargs=3,
                        metavar=('json', 'name', 'order'),
                        default=['logs', 'eval_epoch.accuracy_top1', 'max'])
    parser.add_argument('-l', '--last', action='store_true')
    args = parser.parse_args()
    main(args)
