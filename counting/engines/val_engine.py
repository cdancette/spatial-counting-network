import math
import time
import torch
import datetime
from bootstrap.lib import utils
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.engines.logger import LoggerEngine

def factory():
    return TrainValTestEngine()

class TrainValTestEngine(LoggerEngine):
    """Contains training and evaluation procedures
    """

    def __init__(self):
        super().__init__()
        self.register_hook('validation_on_flush', self.generate_view)


    def train(self):
        """ Launch training procedures

            List of the hooks:
            
            - train_on_start: before the full training procedure

        """
        Logger()('Launching training procedures')

        self.hook('train_on_start')
        while self.epoch < Options()['engine']['nb_epochs']:
            self.train_epoch(self.model, self.dataset['train'], self.optimizer, self.epoch)

            if Options()['dataset'].get('validation_split', None):
                out = self.eval_epoch(self.model, self.dataset['validation'], self.epoch, mode="validation")
            if Options()['dataset']['eval_split']:
                out = self.eval_epoch(self.model, self.dataset['eval'], self.epoch, mode="eval")
            if 'saving_criteria' in Options()['engine'] and Options()['engine']['saving_criteria'] is not None:
                for saving_criteria in Options()['engine']['saving_criteria']:
                    if self.is_best(out, saving_criteria):
                        name = saving_criteria.split(':')[0]
                        Logger()('Saving best checkpoint for strategy {}'.format(name))
                        self.save(Options()['exp']['dir'], 'best_{}'.format(name), self.model, self.optimizer)

            Logger()('Saving last checkpoint')
            self.save(Options()['exp']['dir'], 'last', self.model, self.optimizer)
            self.epoch += 1

        Logger()('Ending training procedures')


    def eval_epoch(self, model, dataset, epoch, mode='eval', logs_json=True):
        """ Launch evaluation procedures for one epoch

            List of the hooks (``mode='eval'`` by default):

            - mode_on_start_epoch: before the evaluation procedure for an epoch
            - mode_on_start_batch: before the evaluation precedure for a batch
            - mode_on_forward: after the forward of the model
            - mode_on_print: after the print to the terminal
            - mode_on_end_batch: end of the evaluation procedure for a batch
            - mode_on_end_epoch: before saving the logs in logs.json
            - mode_on_flush: end of the evaluation procedure for an epoch

            Returns:
                out(dict): mean of all the scalar outputs of the model, indexed by output name, for this epoch
        """
        utils.set_random_seed(Options()['misc']['seed'] + epoch) # to be able to reproduce exps on reload
        Logger()('Evaluating model on {}set for epoch {}'.format(dataset.split, epoch))
        model.eval()
        model.set_mode(mode)

        timer = {
            'begin': time.time(),
            'elapsed': time.time(),
            'process': None,
            'load': None,
            'run_avg': 0
        }
        out_epoch = {}
        batch_loader = dataset.make_batch_loader()

        self.hook('{}_on_start_epoch'.format(mode))
        for i, batch in enumerate(batch_loader):
            timer['load'] = time.time() - timer['elapsed']
            self.hook('{}_on_start_batch'.format(mode))

            with torch.no_grad():
                out = model(batch)
            
            
                if 'loss' in out and torch.isnan(out['loss']):
                    torch.cuda.synchronize()
                    del out
                    Logger()('NaN detected')
                    import gc;gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    continue

            self.hook('{}_on_forward'.format(mode))

            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value('{}_batch.batch'.format(mode), i, should_print=False)
            Logger().log_value('{}_batch.epoch'.format(mode), epoch, should_print=False)
            Logger().log_value('{}_batch.timer.process'.format(mode), timer['process'], should_print=False)
            Logger().log_value('{}_batch.timer.load'.format(mode), timer['load'], should_print=False)

            for key, value in out.items():
                if torch.is_tensor(value):
                    if value.dim() <= 1:
                        value = value.item() # get number from a torch scalar
                    else:
                        continue
                if type(value) == list:
                    continue
                if type(value) == dict:
                    continue
                if value is None:
                    continue
                if key not in out_epoch:
                    out_epoch[key] = []
                out_epoch[key].append(value)
                Logger().log_value('{}_batch.{}'.format(mode, key), value, should_print=False)

            if i % Options()['engine']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{}  elapsed: {} | left: {}".format(' '*len(mode), 
                    datetime.timedelta(seconds=math.floor(time.time() - timer['begin'])),
                    datetime.timedelta(seconds=math.floor(timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{}  process: {:.5f} | load: {:.5f}".format(' '*len(mode), timer['process'], timer['load']))
                self.hook('{}_on_print'.format(mode))
            
            timer['elapsed'] = time.time()
            self.hook('{}_on_end_batch'.format(mode))

            if Options()['engine']['debug']:
                if i > 10:
                    break

        out = {}
        for key, value in out_epoch.items():
            try:
                out[key] = sum(value)/len(value)
            except:
                import ipdb; ipdb.set_trace()

        Logger().log_value('{}_epoch.epoch'.format(mode), epoch, should_print=True)
        for key, value in out.items():
            Logger().log_value('{}_epoch.{}'.format(mode, key), value, should_print=True)

        self.hook('{}_on_end_epoch'.format(mode))
        if logs_json:
            Logger().flush()

        self.hook('{}_on_flush'.format(mode))
        return out
