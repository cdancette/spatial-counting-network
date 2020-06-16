from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.mlp import MLP

from .utils import mask_softmax
from .counter.counter import Counter


class BaselineCountingRegression(nn.Module):
    def __init__(self,
            txt_enc={},
            self_q_att=False,
            agg={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={},
            fusion={},
            residual=False,
            use_counter=False,
            ):
        super().__init__()
        self.self_q_att = self_q_att
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean', 'sum']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.fusion = fusion
        self.residual = residual
        self.use_counter = use_counter
        
        # Modules
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.fusion_module = block.factory_fusion(self.fusion)
        self.classif_module = MLP(**self.classif['mlp'])

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)

      
    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network. 
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')
        bsize = v.shape[0]
        n_regions = v.shape[1]
        # breakpoint()
        out = {}

        q = self.process_question(q, l,)
        out['q_emb'] = q
        q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize*n_regions, -1)

        mm = self.process_fusion(q_expand, v,)  # B * num_regions * dim_v

        # self attention over image: TODO
        if self.residual:
            mm = v + mm

        mm_argmax = None
        if self.agg['type'] == 'max':
            mm, mm_argmax = torch.max(mm, 1)
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)
        elif self.agg['type'] == 'sum':
            mm = mm.sum(dim=1)

        out['mm'] = mm
        out['mm_argmax'] = mm_argmax

        preds = self.classif_module(mm)
        out['pred'] = preds  # B * 1  # backprop on this

        # compute answers
        answers = torch.round(preds).squeeze(1)  # (B,)
        # Logger()(answers)  # TODO probleme ici
        out['logits'] = torch.zeros(bsize, len(self.ans_to_aid))

        for i in range(bsize):
            answer = str(int(answers[i].item()))
            if answer not in self.ans_to_aid:
                answer = '0' # TODO: can we find the closest answer instead maybe ? 
            aid = self.ans_to_aid[answer]  # ugly.. clean this va-et-viens between int and str
            out['logits'][i][aid] = 1.0
        # Logger()(out['logits'])
        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1
        q_emb = txt_enc.embedding(q)

        q, _ = txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att*q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:,0])
            q = txt_enc._select_last(q, l)
        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        # pred = out['logits']
        _, pred = out['logits'].data.max(1)

        pred.squeeze_()
        if batch_size != 1:
            out[f'answers'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids'] = [pred.item()]
        return out


class BaselineCounter(nn.Module):
    def __init__(self,
            txt_enc={},
            self_q_att=False,
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={},
            fusion={},
            ):
        super().__init__()
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.fusion = fusion
        
        # Modules
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.fusion_module = block.factory_fusion(self.fusion)

        self.counter = Counter(objects=36)
        self.mlp_mm = nn.Sequential(
            nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 2048)
        )

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)


    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network. 
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')
        bsize = v.shape[0]
        n_regions = v.shape[1]

        out = {}

        q = self.process_question(q, l,)
        out['q_emb'] = q
        q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize*n_regions, -1)

        mm = self.process_fusion(q_expand, v,)  # B * num_regions * dim_v
        mm = self.mlp_mm(mm)
        att = mm.sum(2)  # B * num_regions

        assert att.size(0) == bsize
        assert att.size(1) == 36
        assert len(att.shape) == 2
        # breakpoint()

        pred, _ = self.counter(batch['coord'].transpose(2, 1), att)
        out['pred'] = pred  # B * num_regions  # backprop on this

        # compute answers
        answers = torch.round(pred).squeeze(1)  # (B,)
        # Logger()(answers)  # TODO probleme ici
        out['logits'] = torch.zeros(bsize, len(self.ans_to_aid))

        for i in range(bsize):
            answer = str(int(answers[i].item()))
            if answer not in self.ans_to_aid:
                if answers[i].item() < 0:
                    answer = '0' # TODO: can we find the closest answer instead maybe ? 
                else:
                    answer = str(Options()['dataset.max_number'])
            aid = self.ans_to_aid[answer]  # ugly.. clean this va-et-viens between int and str
            out['logits'][i][aid] = 1.0
        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1
        q_emb = txt_enc.embedding(q)

        q, _ = txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att*q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:,0])
            q = txt_enc._select_last(q, l)
        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        # pred = out['logits']
        _, pred = out['logits'].data.max(1)

        pred.squeeze_()
        if batch_size != 1:
            out[f'answers'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids'] = [pred.item()]
        return out





class BaselineTally(nn.Module):
    def __init__(self,
            txt_enc={},
            self_q_att=False,
            agg={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={},
            fusion={},
            residual=False,
            use_counter=False,
            ):
        super().__init__()
        self.self_q_att = self_q_att
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean', 'sum']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.fusion = fusion
        self.residual = residual
        self.use_counter = use_counter
        
        # Modules
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.fusion_module = block.factory_fusion(self.fusion)
        self.classif_module = MLP(**self.classif['mlp'])

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)

      
    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network. 
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')
        bsize = v.shape[0]
        n_regions = v.shape[1]
        # breakpoint()
        out = {}

        q = self.process_question(q, l,)
        out['q_emb'] = q
        q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize*n_regions, -1)

        mm = self.process_fusion(q_expand, v,)  # B * num_regions * dim_v

        # self attention over image: TODO
        if self.residual:
            mm = v + mm

        mm_argmax = None
        if self.agg['type'] == 'max':
            mm, mm_argmax = torch.max(mm, 1)
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)
        elif self.agg['type'] == 'sum':
            mm = mm.sum(dim=1)

        out['mm'] = mm
        out['mm_argmax'] = mm_argmax

        preds = self.classif_module(mm)
        out['pred'] = preds  # B * 1  # backprop on this

        # compute answers
        answers = torch.round(preds).squeeze(1)  # (B,)
        # Logger()(answers)  # TODO probleme ici
        out['logits'] = torch.zeros(bsize, len(self.ans_to_aid))

        for i in range(bsize):
            answer = int(answers[i].item())
            if answer < 0:
                answer = 0
            if answer > 15:
                answer = 15
            answer = str(answer)
            aid = self.ans_to_aid[answer]  # ugly.. clean this va-et-viens between int and str
            out['logits'][i][aid] = 1.0
        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1
        q_emb = txt_enc.embedding(q)

        q, _ = txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att*q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:,0])
            q = txt_enc._select_last(q, l)
        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        # pred = out['logits']
        _, pred = out['logits'].data.max(1)

        pred.squeeze_()
        if batch_size != 1:
            out[f'answers'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids'] = [pred.item()]
        return out
