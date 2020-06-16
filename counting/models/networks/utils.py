import torch


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
