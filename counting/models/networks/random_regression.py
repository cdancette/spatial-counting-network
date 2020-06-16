import torch.nn as nn
import torch


class RandomRegression(nn.Module):
    def __init__(
        self,
        aid_to_ans=None,
        ans_to_aid=None,
        wid_to_word=None,
        word_to_wid=None,
        maxn=10,
        always_0=False,
        always_1=False,
        always_2=False,
    ):
        super().__init__()
        self.params = nn.Linear(1, 1)  # to avoid bug
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.maxn = maxn
        self.always_1 = always_1
        self.always_2 = always_2
        self.always_0 = always_0
    def forward(self, batch):
        bsize = batch["visual"].size(0)
        pred = torch.randint(0, self.maxn + 1, (bsize, 1)).float()
        if self.always_0:
            pred.fill_(0.0)
        if self.always_1:
            pred.fill_(1.0)
        if self.always_2:
            pred.fill_(2.0)
        out = {
            "pred": pred,
        }

        answers = pred
        out["logits"] = torch.zeros(bsize, len(self.ans_to_aid))
        for i in range(bsize):
            answer = str(int(answers[i].item()))
            if answer not in self.ans_to_aid:
                answer = "0"  # TODO: can we find the closest answer instead maybe ?
            aid = self.ans_to_aid[
                answer
            ]  # ugly.. clean this va-et-viens between int and str
            out["logits"][i][aid] = 1.0
        return out

    def process_answers(self, out):
        batch_size = out["logits"].shape[0]
        # pred = out['logits']
        _, pred = out["logits"].data.max(1)

        pred.squeeze_()
        if batch_size != 1:
            out[f"answers"] = [
                self.aid_to_ans[pred[i].item()] for i in range(batch_size)
            ]
            out[f"answer_ids"] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f"answers"] = [self.aid_to_ans[pred.item()]]
            out[f"answer_ids"] = [pred.item()]
        return out
