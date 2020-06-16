import torch
import torch.nn as nn
from counting.models.networks.utils import pred_to_logits


class ImageOnly(nn.Module):
    def __init__(
        self,
        dim_v=2048,
        max_ans=15,
        output="classif",
        aid_to_ans=None,
        ans_to_aid=None,
        wid_to_word=None,
        word_to_wid=None,
    ):
        super().__init__()
        self.output = output
        self.max_ans = max_ans
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        if self.output == "classif":
            self.classifier = nn.Sequential(
                nn.Linear(dim_v, dim_v), nn.ReLU(), nn.Linear(dim_v, len(aid_to_ans))
            )
        elif self.output == "regression":
            self.classifier = nn.Sequential(
                nn.Linear(dim_v, dim_v), nn.ReLU(), nn.Linear(dim_v, 1)
            )
        else:
            raise ValueError(self.output)

    def forward(self, batch):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        v = batch["visual"]
        v = v.mean(dim=1)  # b, dim
        output = self.classifier(v)

        if self.output == "classif":
            return {"logits": output}
        elif self.output == "regression":
            return {
                "pred": output,
                "logits": pred_to_logits(
                    output, max_ans=self.max_ans, ans_to_aid=self.ans_to_aid
                ),
            }

    def process_answers(self, out):
        batch_size = out["logits"].shape[0]
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
