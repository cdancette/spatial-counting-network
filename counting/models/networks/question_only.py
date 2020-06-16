import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.vqa_net import mask_softmax
from counting.models.networks.utils import pred_to_logits


class QuestionOnly(nn.Module):
    def __init__(
        self,
        txt_enc={},
        max_ans=15,
        self_q_att=False,
        dim_q=2400,
        wid_to_word={},
        word_to_wid={},
        aid_to_ans=[],
        ans_to_aid={},
        output="classification",
    ):
        super().__init__()
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.max_ans = max_ans
        self.output = output

        if self.output == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(dim_q, 1024), nn.ReLU(), nn.Linear(1024, len(aid_to_ans)),
            )

        elif self.output == "regression":
            self.classifier = nn.Sequential(
                nn.Linear(dim_q, 1024), nn.ReLU(), nn.Linear(1024, 1),
            )

        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(dim_q, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        Logger().log_value(
            "nparams",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True,
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

    def forward(self, batch):
        v = batch["visual"]
        q = batch["question"]
        l = batch["lengths"].data
        q = self.process_question(q, l)

        if self.output == "classification":
            logits = self.classifier(q)
            out = {"logits": logits}
        elif self.output == "regression":
            pred = self.classifier(q)
            out = {
                "pred": pred,
                "logits": pred_to_logits(
                    pred, max_ans=self.max_ans, ans_to_aid=self.ans_to_aid
                ),
            }

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
            q = self.txt_enc._select_last(q, l)

        return q

    def process_answers(self, out):
        batch_size = out["logits"].shape[0]
        _, pred = out["logits"].data.max(1)
        pred.squeeze_()
        out["answers"] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out["answer_ids"] = [pred[i] for i in range(batch_size)]
        return out


class QuestionImage(nn.Module):
    def __init__(
        self,
        txt_enc={},
        dim_q=2400,
        max_ans=15,
        self_q_att=False,
        wid_to_word={},
        word_to_wid={},
        aid_to_ans=[],
        ans_to_aid={},
        output="classification",
    ):
        super().__init__()
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.max_ans = max_ans
        self.output = output

        if self.output == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(2048 + dim_q, 1024),
                nn.ReLU(),
                nn.Linear(1024, len(aid_to_ans)),
            )

        elif self.output == "regression":
            self.classifier = nn.Sequential(
                nn.Linear(2048 + dim_q, 1024), nn.ReLU(), nn.Linear(1024, 1),
            )

        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(dim_q, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        Logger().log_value(
            "nparams",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True,
        )

        Logger().log_value(
            "nparams_txt_enc", self.get_nparams_txt_enc(), should_print=True
        )

        self.buffer = None

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

    def forward(self, batch):
        v = batch["visual"]
        q = batch["question"]
        l = batch["lengths"].data
        q = self.process_question(q, l)

        v = v.mean(dim=1)  # (bsize, dim)
        embedding = torch.cat((q, v), dim=1)

        if self.output == "classification":
            logits = self.classifier(embedding)
            out = {"logits": logits}
        elif self.output == "regression":
            pred = self.classifier(embedding)
            out = {
                "pred": pred,
                "logits": pred_to_logits(pred, max_ans=self.max_ans, ans_to_aid=self.ans_to_aid),
            }

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
            q = self.txt_enc._select_last(q, l)

        return q

    def process_answers(self, out):
        batch_size = out["logits"].shape[0]
        _, pred = out["logits"].data.max(1)
        pred.squeeze_()
        out["answers"] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out["answer_ids"] = [pred[i] for i in range(batch_size)]
        return out
