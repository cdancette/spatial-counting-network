import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from counting.models.networks.utils import pred_to_logits

class Guess(nn.Module):
    def __init__(
        self,
        max_ans=None,
        wid_to_word={},
        word_to_wid={},
        aid_to_ans=[],
        ans_to_aid={},
        answer=1.0,
    ):
        super().__init__()
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.max_ans = max_ans
        self.answer = answer
        self.parameters = nn.Linear(1, 1)  # just for optimizer
        # self.bsize = Options()['dataset.batch_size']

    def forward(self, batch):
        bsize = batch['class_id'].shape[0]
        out = {}

        pred = torch.full((bsize, 1), fill_value=self.answer).float().to(device=batch['class_id'].device)
        out["pred"] = pred
        out["answer"] = pred.round()
        out["logits"] = pred_to_logits(pred, self.max_ans, self.ans_to_aid).to(
            device=pred.device
        )
        return out

    def process_answers(self, out):
        batch_size = out["logits"].shape[0]
        _, pred = out["logits"].data.max(1)
        pred.squeeze_()
        out["answers"] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out["answer_ids"] = [pred[i] for i in range(batch_size)]
        return out