from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.vqa_net import mask_softmax

# from block.models.networks.mlp import MLP
# from counting.models.networks.murel.pairwise import Pairwise
# from counting.models.networks.utils import pred_to_logits
# from counting.models.networks.attention_counting import TransformerBlock, OUTPUTS
from block.models.networks.fusions.fusions import MLB
from counting.models.networks.sharpen_temp import TemperatureSigmoid


def pred_to_logits(pred, max_ans=15, ans_to_aid=None):
    if ans_to_aid is None:
        ans_to_aid = {str(i): i for i in range(16)}
    pred = pred.round()
    num_ans = len(ans_to_aid)
    bsize = pred.shape[0]
    logits = torch.zeros((bsize, num_ans)).to(device=pred.device)
    for i in range(bsize):
        answer = str(int(pred[i].item()))
        if answer not in ans_to_aid:
            if pred[i].item() < 0:
                answer = "0"
            else:
                answer = str(max_ans)
        try:
            aid = ans_to_aid[answer]
        except:
            breakpoint()
        logits[i][aid] = 1.0
    return logits


class TransformerBlock(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        embed_dim,
        num_heads,
        intermediate_dim,
        output_dim=None,
        dropout_attention=0.1,
        dropout_output=0.1,
    ):
        super().__init__()
        self.query = nn.Linear(query_dim, embed_dim)
        self.key = nn.Linear(key_dim, embed_dim)
        self.value = nn.Linear(key_dim, embed_dim)
        self.multi_head_att = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_attention
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, embed_dim),
            nn.Dropout(dropout_output),
        )

    def forward(self, emb_for_query, inputs):
        """
        input1
        """
        query = self.query(emb_for_query).permute(1, 0, 2)  # (length, batch, dim)
        key = self.key(inputs).permute(1, 0, 2)  # (length, batch, dim)
        value = self.value(inputs).permute(1, 0, 2)  # (length, batch, dim)
        attn_output, attn_output_weights = self.multi_head_att(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)  # batch, length, dim
        # if attn_output.shape != emb_for_query.shape:
        #     breakpoint()
        intermediate = self.layernorm1(attn_output + emb_for_query)
        output = self.feedforward(intermediate)
        output = self.layernorm2(intermediate + output)
        return output


class ClassificationOutput(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def forward(self, v_emb, q_emb=None):
        pooled_v = v_emb.mean(dim=1)  # N, dim
        return {"logits": self.logits(pooled_v)}


class NaiveRegressionOutput(nn.Module):
    def __init__(self, hidden_dim, max_ans):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.max_ans = max_ans

    def forward(self, v_emb, q_emb=None):
        pooled_v = v_emb.mean(dim=1)  # N, dim
        pred = self.logits(pooled_v)
        pred.sum(dim=1)
        return {
            "pred": pred,
            "logits": pred_to_logits(pred, self.max_ans),
        }


class SimpleRegressionOutput(nn.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim=None,
        sigmoid=True,
        temperature=False,
        relu=False,
        max_ans=15,
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 2
        self.logits = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, 1),
        )
        self.sigmoid = sigmoid
        self.max_ans = max_ans
        self.relu = relu
        self.temperature = temperature
        # if self.temperature:
        # self.sharpen_temp = TemperatureSigmoid()

    def forward(self, v_emb, q_emb=None):
        scores = self.logits(v_emb)  # N, R, 1
        if self.sigmoid:
            scores = torch.sigmoid(scores)
        elif self.relu:
            scores = torch.relu(scores)
        pred = scores.sum(dim=1)  # (N, 1)

        return {
            "scores": scores,
            "final_attention_map": scores,
            "pred": pred,
            "logits": pred_to_logits(pred, self.max_ans),
        }



class MLBOutputRegression(nn.Module):
    def __init__(
        self,
        input_dim,
        fusion_mm_dim=None,
        sigmoid=True,
        relu=False,
        temperature=False,
        temperature_params={},
        fusion_activ=None,
        max_ans=15,
    ):
        super().__init__()
        self.module = MLB(
            [input_dim, input_dim],
            1,
            mm_dim=fusion_mm_dim,
            activ_input=fusion_activ,
            activ_output=fusion_activ,
            normalize=True,
            dropout_input=0.1,
            dropout_pre_lin=0.0,
            dropout_output=0.0,
        )

        self.sigmoid = sigmoid
        self.max_ans = max_ans
        self.relu = relu
        self.temperature = temperature
        if self.temperature:
            Logger()("using temperature")
            self.temp_sigmoid = TemperatureSigmoid(**temperature_params)


    def forward(self, v_emb, q_emb):
        """
        v_emb: N, R, dim
        q_emb: N, R, dim
        """
        n, r, _ = v_emb.shape

        scores = self.module((v_emb, q_emb))  # (n, r, 1)
        scores = scores.view((n, r, 1))

        if self.temperature:
            scores = self.temp_sigmoid(scores)
        elif self.sigmoid:
            scores = torch.sigmoid(scores)
        elif self.relu:
            scores = torch.relu(scores)

        pred = scores.sum(dim=1)  # (N, 1)

        return {
            "scores": scores,
            "final_attention_map": scores,
            "pred": pred,
            "logits": pred_to_logits(pred, self.max_ans),
        }


class RegressionFusionOutput(nn.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim=None,
        sigmoid=True,
        temperature=False,
        temperature_params={},
        fusion="elementwise",
        layers=2,
        relu=False,
        max_ans=15,
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 2
        self.layers = layers
        if layers == 2:
            self.logits = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, 1),
            )
        elif layers == 1:
            self.logits = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1),
            )

        self.sigmoid = sigmoid
        self.fusion = fusion
        self.max_ans = max_ans
        self.relu = relu
        self.temperature = temperature
        if self.temperature:
            Logger()("using temperature")
            self.temp_sigmoid = TemperatureSigmoid(**temperature_params)

    def forward(self, v_emb, q_emb):
        """
        v_emb: N, R, dim
        q_emb: N, R, dim
        """
        q_emb = q_emb.mean(dim=1)

        if self.fusion == "elementwise":
            fusion = torch.einsum("nrd,nd->nrd", v_emb, q_emb)
            scores = self.logits(fusion)  # N, R, 1
            if self.temperature:
                scores = self.temp_sigmoid(scores)
            elif self.sigmoid:
                scores = torch.sigmoid(scores)
            elif self.relu:
                scores = torch.relu(scores)
        elif self.fusion == "cosine":
            q_emb = F.normalize(q_emb, dim=1)
            v_emb = F.normalize(v_emb, dim=2)
            scores = torch.einsum("nrd,nd->nr", v_emb, q_emb)
            scores = torch.abs(scores)
            scores = scores.view(v_emb.shape[0], v_emb.shape[1], 1)

        pred = scores.sum(dim=1)  # (N, 1)

        return {
            "scores": scores,
            "final_attention_map": scores,
            "pred": pred,
            "logits": pred_to_logits(pred, self.max_ans),
        }


class MultiHeadPairwiseFusionOutput(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        fusion="elementwise",
        sigmoid=True,
        dropout_attention=0.1,
        max_ans=None,
    ):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.pairwie_att = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_attention
        )
        self.fusion = fusion
        self.sigmoid = sigmoid
        self.max_ans = max_ans

        self.layernorm1 = nn.LayerNorm(hidden_dim)

        self.regression = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, v_emb, q_emb):
        """
        here we do:
        - one pairwise vision (with no residual).
            Why no residual: because we don't want to keep too much information. We want to filter
            some out.
        - one final fusion: 
            - take an aggregated question representation
            - dot product
        """
        # pairwise
        query = self.query(v_emb).permute(1, 0, 2)  # (length, batch, dim)
        key = self.key(v_emb).permute(1, 0, 2)  # (length, batch, dim)
        value = self.value(v_emb).permute(1, 0, 2)  # (length, batch, dim)
        attn_output, attn_output_weights = self.pairwie_att(query, key, value)
        pairwise_output = attn_output.permute(1, 0, 2)  # batch, length, dim
        intermediate = self.layernorm1(pairwise_output)  # b, length, dim

        # final fusion
        q_emb = q_emb.mean(dim=1)
        if self.fusion == "elementwise":
            fusion = torch.einsum("nrd,nd->nrd", intermediate, q_emb)
        else:
            raise ValueError(self.fusion)

        scores = self.regression(fusion)  # N, R, 1

        if self.sigmoid:
            scores = torch.sigmoid(scores)
        elif self.relu:
            scores = torch.relu(scores)
        pred = scores.sum(dim=1)  # (N, 1)

        return {
            "scores": scores,
            "final_attention_map": scores,
            "pred": pred,
            "logits": pred_to_logits(pred, self.max_ans),
        }


class CounterOutput(nn.Module):
    def __init__(self, hidden_dim, max_ans=15):
        super().__init__()
        from counting.models.networks.counter.counter import Counter

        self.logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.max_ans = max_ans
        self.counter = Counter(36, already_sigmoided=True)

    def forward(self, v_emb, boxes=None):
        scores = self.logits(v_emb)  # N, R, 1
        scores = torch.sigmoid(scores)
        pred, _ = self.counter(boxes.transpose(2, 1), scores.squeeze(2))

        return {
            "scores": scores,
            "final_attention_map": scores,
            "pred": pred,
            "logits": pred_to_logits(pred, self.max_ans),
        }


OUTPUTS = {
    "classification": ClassificationOutput,
    "naive-regression": NaiveRegressionOutput,
    "simple-regression": SimpleRegressionOutput,
    "fusion-regression": RegressionFusionOutput,
    "pairwise-fusion-output": MultiHeadPairwiseFusionOutput,
    "counter": CounterOutput,
    "mlb-regression": MLBOutputRegression,
}


class AttentionCountingMLB(nn.Module):
    def __init__(
        self,
        txt_enc={},
        self_q_att=False,
        max_ans=None,
        # answer_scaling=False,
        # model
        n_steps=1,
        add_coords=True,
        share_question_cells=False,
        share_vision_cells=False,
        multiple_question_cells=False,
        recursive_question_cells=True,
        self_attention_vision=True,
        fusion_vision=True,
        num_heads_self_att=None,
        intermediate_dim_self_att=None,
        hidden_dim=2048,
        # fusion
        fusion_mm_dim=1200,
        fusion_activ="relu",
        attention_scaling=1.0,
        residual_fusion=False,
        # output
        output=None,
        output_on="final",
        output_params={},
        wid_to_word={},
        word_to_wid={},
        aid_to_ans=[],
        ans_to_aid={},
        engine=None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.max_ans = max_ans
        self.attention_scaling = attention_scaling
        self.add_coords = add_coords
        # self.

        # self.branches = branches

        self.share_vision_cells = share_vision_cells
        self.share_question_cells = share_question_cells
        self.recursive_question_cells = recursive_question_cells
        self.multiple_question_cells = multiple_question_cells
        self.self_attention_vision = self_attention_vision
        self.fusion_vision = fusion_vision
        self.residual_fusion = residual_fusion

        self.output_on = output_on
        self.output = output
        self.output_params = output_params

        # if self.add_lvis:
        #     self.lin_lvis = nn.Linear(1024, 2048)

        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.q_projection = nn.Linear(2400, hidden_dim)

        self.hidden_dim = hidden_dim
        if hidden_dim != 2048:
            self.v_projection = nn.Linear(2048, hidden_dim)

        def transformer_block():
            return TransformerBlock(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                num_heads=num_heads_self_att,
                intermediate_dim=intermediate_dim_self_att,
            )

        def fusion_block():
            return MLB(
                [hidden_dim, hidden_dim],
                hidden_dim,
                mm_dim=fusion_mm_dim,
                activ_input=fusion_activ,
                activ_output=fusion_activ,
                normalize=True,
                dropout_input=0.1,
                dropout_pre_lin=0.0,
                dropout_output=0.0,
            )

        if self.share_vision_cells:
            self.att_vis_block = transformer_block()
            # copy weights
            if self.self_attention_vision:
                self.self_attention_vision_block = [self.att_vis_block] * self.n_steps
            # bi attention, copy weights
            if self.fusion_vision:
                self.vi_fusion_block = fusion_block()
                self.vi_fusion_blocks = [self.vi_fusion_block] * self.n_steps
        else:
            # different modules
            if self.self_attention_vision:
                self.self_attention_vision_block = nn.ModuleList(
                    [transformer_block() for _ in range(self.n_steps)]
                )
                Logger().log_value(
                    "self-att-vision.nparams",
                    self.get_nparams(self.self_attention_vision_block),
                    should_print=True,
                )
            if self.fusion_vision:
                self.vi_fusion_blocks = nn.ModuleList(
                    [fusion_block() for _ in range(self.n_steps)]
                )
                Logger().log_value(
                    "vi_fusion_blocks.nparams",
                    self.get_nparams(self.vi_fusion_blocks),
                    should_print=True,
                )

        # text
        if self.multiple_question_cells:
            if self.share_question_cells:
                # copy
                self.question_cell = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                )
                self.question_cells = [self.question_cell] * self.n_steps
            else:
                self.question_cells = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                        for _ in range(self.n_steps)
                    ]
                )

        # output module
        # self.
        self.output_module = OUTPUTS[self.output](**output_params)

        # if self.output_on == "initial+final":
        #     self.output_module_initial

        if output == "fusion-regression" and output_params.get("temperature", False):
            Logger()("Registering hook for train")
            engine.register_hook(
                "train_on_end_epoch", self.output_module.temp_sigmoid.step
            )
        Logger().log_value(
            "output.nparams", self.get_nparams(self.output_module), should_print=True,
        )

        if self.add_coords:
            self.lin_coords = nn.Linear(4, hidden_dim)

        Logger().log_value(
            "nparams", self.get_nparams(self), should_print=True,
        )

        Logger().log_value(
            "nparams_txt_enc", self.get_nparams_txt_enc(), should_print=True
        )

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [
                p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad
            ]
            params += [
                p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad
            ]
        return sum(params)

    def get_nparams(self, module):
        params = [p.numel() for p in module.parameters() if p.requires_grad]
        return sum(params)

    def forward(self, batch):
        v = batch["visual"]  # b, n, dim
        q = batch["question"]
        l = batch["lengths"].data
        coords = batch["norm_coord"]

        bsize = q.shape[0]
        # n_regions = v.shape[1]
        out = {}

        if self.hidden_dim != 2048:
            v = self.v_projection(v)

        if self.add_coords:
            v = v + self.lin_coords(coords)

        q = self.process_question(q, l)  # b, dim
        q = self.q_projection(q)

        initial_v_emb = v
        initial_q_emb = q

        for i in range(self.n_steps):
            # fusion
            if self.multiple_question_cells:
                if self.recursive_question_cells:
                    q = self.question_cells[i](q)
                else:
                    q = self.question_cells[i](initial_q_emb)

            # fusion
            v_before = v
            shape_v = v.shape  # b, regions, dim
            q_expand = q[:, None, :].expand_as(v)  # b, regions, dim
            dim = v.shape[-1]
            q_expand = q_expand.reshape(-1, dim)
            v = v.view(-1, dim)

            if self.fusion_vision:
                v = self.vi_fusion_blocks[i]((v, q_expand))
            v = v.view(shape_v)
            if self.residual_fusion:
                v = v + v_before
            if self.self_attention_vision:
                v = self.self_attention_vision_block[i](v, v)

        # output
        if self.output_on == "final":
            # if self.output == "counter":
            #     return self.output_module(v_emb, boxes=coords)
            return self.output_module(v, q[:, None, :])

        if self.output_on == "initial+final":
            if self.output == "counter":
                raise NotImplementedError()
            else:
                out_initial = self.output_module(
                    initial_v_emb, initial_q_emb[:, None, :]
                )
                out_final = self.output_module(v, q[:, None, :])
            scores = out_initial["scores"] * out_final["scores"]
            pred = scores.sum(dim=1)  # (N, 1)
            return {
                "scores": scores,
                "final_attention_map": scores,
                "pred": pred,
                "logits": pred_to_logits(pred, self.max_ans, self.ans_to_aid),
            }
        else:
            raise ValueError(self.output_on)

        return out

    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q, _ = self.txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = self.q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = self.q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            # self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att * q
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
            l = list(l.data[:, 0])
            try:
                q = self.txt_enc._select_last(q, l)
            except:
                breakpoint()
        return q

    def process_answers(self, out):
        batch_size = out["logits"].shape[0]
        _, pred = out["logits"].data.max(1)
        pred.squeeze_()
        out["answers"] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out["answer_ids"] = [pred[i] for i in range(batch_size)]
        return out
