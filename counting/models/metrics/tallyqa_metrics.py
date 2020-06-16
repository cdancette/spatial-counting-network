import torch
import torch.nn as nn


import numpy as np
import os
import traceback
import scipy.stats
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from bootstrap.models.metrics.accuracy import accuracy
from bootstrap import Logger, Options
from counting.models.metrics.mean_ap import DetectionMAP


IOU_COLUMNS = [
    "question_id",
    "name",
    "answer",  # gt answer
    "gt_bboxes",
    "pred",  # model prediction
    "pred_round",  # rounded prediction
    "candidate_bbox",  # [(box, score)] of faster rcnn (score by model, sum(scores) = pred)
    "iou",  # global iou between bbox and scores
    "iou_sum",  # global iou on pixels(sum then threshold)
    "ioo",  # global ioo (intersection over ours) == precision
    "ioo_sum",
    "iogt",  # global iogt (intersection over gt) == recall
    "iogt_sum",
    "iou_boxes",  # sum of weighted iou (by box score) (must be normalized)
    "iou_boxes_norm",  # normalized weighted iou
    "ioo_boxes",  # weighted ioo (must be normalized) == similar to IRLC metric
    "ioo_boxes_norm",  # normalized weighted ioo
    "iogt_boxes",  # weighted iogt
    "iogt_boxes_norm",
]

THRESHOLDS_mAP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TallyQAMetrics(nn.Module):
    def __init__(self, mode, engine, topk=[1, 5]):
        super().__init__()
        self.mode = mode
        self.engine = engine
        self.topk = topk
        self.ans_to_aid = self.engine.dataset[mode].ans_to_aid

        engine.register_hook("{}_on_start_epoch".format(mode), self.reset)
        engine.register_hook("{}_on_end_epoch".format(mode), self.compute_accuracy)

        self.exp_dir = Options()["exp.dir"]
        self.score_threshold_grounding = Options()["model.metric"].get(
            "score_threshold_grounding", 0.5
        )

    def reset(self):
        self.ious = {"overall": []}
        self.ious_nonzero = {"overall": []}
        self.ious_sum = {"overall": []}
        self.ious_sum_nonzero = {"overall": []}
        self.all_ious = []  # question_id, answer, object, iou, iou_sum
        for number in range(16):
            self.ious[number] = []
            self.ious_nonzero[number] = []
            self.ious_sum[number] = []
            self.ious_sum_nonzero[number] = []

        self.answers = dict()
        for t in ("simple", "complex", "overall", "positional", "type"):
            self.answers[t] = {"ans": [], "pred": [], "gt": [], "hard.ans": []}
            for number in range(16):
                key = f"{t}-{number}"
                self.answers[key] = {"ans": [], "pred": [], "gt": [], "hard.ans": []}
            for subcat in ("own", "opposite"):
                key = f"{t}-{subcat}"
                self.answers[key] = {"ans": [], "pred": [], "gt": [], "hard.ans": []}
        for t in ("even", "odd"):
            self.answers[t] = {"ans": [], "pred": [], "gt": [], "hard.ans": []}

        self.mean_ap = dict()
        for t in THRESHOLDS_mAP:
            self.mean_ap[t] = DetectionMAP(
                n_class=1, pr_samples=11, overlap_threshold=t
            )

    def compute_accuracy(self):
        accs = {}
        rmses = {}
        for t in self.answers.keys():
            # for t in ("simple", "complex", "overall", "overall_same", "overall_diff", "overall_same_diff"):
            if len(self.answers[t]["gt"]) > 0:
                gt = torch.tensor(self.answers[t]["gt"]).long()
                if len(self.answers[t]["pred"]) > 0:
                    pred = torch.tensor(self.answers[t]["pred"])
                    acc = (pred.round().long() == gt).sum().float() / len(pred)
                else:
                    pred = torch.tensor(self.answers[t]["ans"])
                    acc = (pred.long() == gt).sum().float() / len(pred)
                accs[t] = acc
                Logger().log_value(
                    f"{self.mode}_epoch.tally_acc.{t}",
                    acc.item() * 100,
                    should_print=True,
                )

                # compute L1, L2, RMSE
                diff = (pred - gt).float()
                l1 = torch.abs(diff).mean()
                Logger().log_value(f"{self.mode}_epoch.tally_l1.{t}", l1.item())
                l2 = (diff ** 2).mean()
                Logger().log_value(f"{self.mode}_epoch.tally_l2.{t}", l2.item())
                rmse = torch.sqrt(l2)
                rmses[t] = rmse.item()
                Logger().log_value(
                    f"{self.mode}_epoch.tally_rmse.{t}", rmse.item(), should_print=True
                )

                # compute score with hard threshold
                if len(self.answers[t]["hard.ans"]) > 0:
                    hard_pred = torch.tensor(self.answers[t]["hard.ans"]).long()
                    acc = (hard_pred == gt).sum().float() / len(hard_pred)
                    Logger().log_value(
                        f"{self.mode}_epoch.tally_thresh_acc.{t}",
                        acc.item() * 100,
                        should_print=True,
                    )

        # compute mean acc complex and simple
        if "simple" in accs and "complex" in accs:
            mean_acc = (accs["simple"].item() + accs["complex"].item()) / 2
            Logger().log_value(
                f"{self.mode}_epoch.tally_acc.mean_complex_simple",
                mean_acc * 100,
                should_print=True,
            )

        # compute normalized accuracy
        for t in ("overall", "simple", "complex", "positional", "type"):

            accs_t = [
                accs[f"{t}-{number}"] for number in range(16) if f"{t}-{number}" in accs
            ]
            if accs_t:
                # normalized arithmetic accuracy
                m_rel_acc = np.mean(accs_t)
                Logger().log_value(
                    f"{self.mode}_epoch.tally_acc.m-rel.{t}",
                    m_rel_acc.item() * 100,
                    should_print=True,
                )

            # normalized harmonic accuracy
            try:
                normalized_harmonic_acc = scipy.stats.hmean(accs_t).item()
            except ValueError:
                # there is a zero value in list
                normalized_harmonic_acc = 0.0
            Logger().log_value(
                f"{self.mode}_epoch.tally_acc.norm_harmonic.{t}",
                normalized_harmonic_acc * 100,
            )

            # normalized RMSE
            rmses_t = [
                rmses[f"{t}-{number}"]
                for number in range(16)
                if f"{t}-{number}" in rmses
            ]
            if rmses_t:
                m_rel_rmse = np.mean(rmses_t)
                Logger().log_value(
                    f"{self.mode}_epoch.tally_rmse.m-rel.{t}", m_rel_rmse.item()
                )

            # normalized
            try:
                normalized_harmonic_rmse = scipy.stats.hmean(rmses_t).item()
            except ValueError:
                # zero value (should not happen for regression)
                normalized_harmonic_rmse = 0
            Logger().log_value(
                f"{self.mode}_epoch.tally_rmse.norm_harmonic.{t}",
                normalized_harmonic_rmse,
            )

        # Metrics for COCO grounding
        if self.all_ious:
            all_ious = pd.DataFrame(self.all_ious, columns=IOU_COLUMNS)
            exp_dir = self.exp_dir
            threshold = self.score_threshold_grounding
            iou_path = os.path.join(exp_dir, f"iou-{threshold}.pickle")
            all_ious.to_pickle(iou_path)
            # log other metrics
            columns_to_log = [
                "iou",
                "iou_sum",
                "ioo",
                "ioo_sum",
                "iogt",
                "iogt_sum",
                "iou_boxes_norm",
                "ioo_boxes_norm",
                "iogt_boxes_norm",
            ]
            iou_nonzero = all_ious[all_ious["answer"] != 0]
            for metric in columns_to_log:
                # overall
                m = all_ious[metric].mean()
                Logger().log_value(
                    f"{self.mode}_epoch.{metric}.overall", m, should_print=True
                )
                m = iou_nonzero[metric].mean()
                Logger().log_value(
                    f"{self.mode}_epoch.{metric}.overall.nonzero", m, should_print=True
                )
                # by number
                # by object
                for i in range(16):
                    m = all_ious[all_ious["answer"] == i][metric].mean()
                    Logger().log_value(
                        f"{self.mode}_epoch.{metric}.{i}", m, should_print=True
                    )

                # by object
                for name in all_ious.name.unique():
                    m = all_ious[all_ious.name == name][metric].mean()
                    Logger().log_value(
                        f"{self.mode}_epoch.{metric}.{name}", m, should_print=False
                    )
                    m = iou_nonzero[iou_nonzero.name == name][metric].mean()
                    Logger().log_value(
                        f"{self.mode}_epoch.{metric}.{name}.nonzero",
                        m,
                        should_print=False,
                    )

            # additionally, do global normalization
            for metric in ["iou_boxes", "ioo_boxes", "iogt_boxes"]:
                m = all_ious[metric].sum() / all_ious["pred"].sum()
                Logger().log_value(
                    f"{self.mode}_epoch.{metric}.overall", m, should_print=True
                )
                m = iou_nonzero[metric].sum() / iou_nonzero["pred"].sum()
                Logger().log_value(
                    f"{self.mode}_epoch.{metric}.overall.nonzero", m, should_print=True
                )

                for i in range(16):
                    all_ious_i = all_ious[all_ious["answer"] == i]
                    m = all_ious_i[metric].sum() / all_ious_i["pred"].sum()
                    Logger().log_value(
                        f"{self.mode}_epoch.{metric}.{i}", m, should_print=True
                    )

                for name in all_ious.name.unique():
                    all_ious_name = all_ious[all_ious.name == name]
                    m = all_ious_name[metric].sum() / all_ious_name["pred"].sum()
                    Logger().log_value(
                        f"{self.mode}_epoch.{metric}.{name}", m, should_print=False
                    )
                    ious_nz_name = iou_nonzero[iou_nonzero.name == name]
                    m = ious_nz_name[metric].sum() / ious_nz_name["pred"].sum()
                    Logger().log_value(
                        f"{self.mode}_epoch.{metric}.{name}.nonzero",
                        m,
                        should_print=False,
                    )

            # mean average precision
            for t in THRESHOLDS_mAP:
                scores = self.mean_ap[t].get_scores()
                # breakpoint()
                Logger().log_value(
                    f"{self.mode}_epoch.mAP.{t}.overall", scores, should_print=True
                )

    def forward(self, cri_out, net_out, batch):
        out = {}
        logits = net_out["logits"].data.cpu()
        class_id = batch["class_id"]
        acc_out = accuracy(logits, class_id.data.cpu(), topk=self.topk)

        for i, k in enumerate(self.topk):
            out["accuracy_top{}".format(k)] = acc_out[i]

        # compute accuracy on simple and difficult examples
        answers = torch.argmax(logits, dim=1)

        for i in range(len(net_out["logits"])):
            pred = answers[i].item()
            gt = batch["answer"][i]

            categories = {"overall"}
            if "issimple" in batch and batch["issimple"][i]:
                main_cat = "simple"
                categories.add("simple")
            elif "issimple" in batch and not batch["issimple"][i]:
                main_cat = "complex"
                categories.add("complex")
            else:
                main_cat = None

            # add categories per number
            categories.add(f"overall-{gt}")
            # if "simple" in categories:
            if main_cat is not None:
                categories.add(f"{main_cat}-{gt}")
            # elif "complex" in categories:
            #     categories.add(f"complex-{gt}")

            if any(
                word in batch["raw_question"][i]
                for word in ["left of", "right of", "behind", "front of",]
            ):
                categories.add("positional")

            if any(word in batch["raw_question"][i] for word in ["type", "types"]):
                categories.add("type")

            if int(batch["answer"][i]) % 2 == 0:
                categories.add("even")
            if int(batch["answer"][i]) % 2 == 1:
                categories.add("odd")

            if hasattr(self.engine.dataset[self.mode], "own_numbers"):
                own_numbers = self.engine.dataset[self.mode].own_numbers
                opposite_numbers = self.engine.dataset[self.mode].opposite_numbers
                if int(batch["answer"][i]) in own_numbers:
                    categories.add("overall-own")
                    if main_cat is not None:
                        categories.add(f"{main_cat}-own")
                if int(batch["answer"][i]) in opposite_numbers:
                    categories.add("overall-opposite")
                    if main_cat is not None:
                        categories.add(f"{main_cat}-opposite")

            for cat in categories:
                if "pred" in net_out:
                    self.answers[cat]["pred"].append(net_out["pred"][i].item())
                self.answers[cat]["ans"].append(pred)
                self.answers[cat]["gt"].append(gt)

                if "final_attention_map" in net_out:
                    thresh_prediction = (net_out["final_attention_map"][i] > 0.5).sum()
                    # breakpoint()
                    self.answers[cat]["hard.ans"].append(thresh_prediction.item())

        # GROUNDING
        if "scores" in net_out and "gt_bboxes" in batch:
            Logger()("Computing COCO grounding")
            bsize = logits.shape[0]
            # compute grounding
            ious = []
            threshold = self.score_threshold_grounding
            for i in range(bsize):
                gt = batch["answer"][i]
                scores = net_out["scores"][i]  # (regions, 1)
                selection = (scores >= threshold).view((scores.shape[0],))
                coords = batch["coord"][i]
                coord_thresh = coords[selection]
                iou, inter, union, ioo, iogt = compute_iou(
                    batch["gt_bboxes"][i], coord_thresh.cpu().numpy()
                )

                ious.append(iou)
                self.ious["overall"].append(iou)
                self.ious[gt].append(iou)
                if batch["answer"][i] != 0:
                    self.ious_nonzero["overall"].append(iou)
                    self.ious_nonzero[gt].append(iou)

                # try another method
                width = batch["img_width"][i]
                height = batch["img_height"][i]
                img_gt = np.full((width, height), False, dtype=bool)  # (x, y)
                img_proposed = np.zeros((width, height))
                for bbox in batch["gt_bboxes"][i]:
                    x, y, x2, y2 = [round(x) for x in bbox]
                    img_gt[x:x2, y:y2] = True
                scores = net_out["scores"][i]
                candidate_bbox = list(
                    zip(
                        batch["coord"][i].tolist(),
                        scores.view((scores.shape[0],)).cpu().tolist(),
                    )
                )

                for bbox, score in candidate_bbox:
                    x, y, x2, y2 = [round(x) for x in bbox]
                    img_proposed[x:x2, y:y2] += score
                thresh = img_proposed >= threshold
                intersection = thresh & img_gt
                union = thresh | img_gt
                union_sum = union.sum()
                inter_sum = intersection.sum()
                thresh_sum = thresh.sum()
                img_gt_sum = img_gt.sum()

                if union_sum == 0:
                    iou_sum = 1.0
                else:
                    iou_sum = inter_sum / union_sum
                if thresh_sum != 0:
                    ioo_sum = inter_sum / thresh_sum
                else:
                    ioo_sum = 1.0

                if img_gt_sum != 0:
                    iogt_sum = inter_sum / img_gt_sum
                else:
                    iogt_sum = 1.0

                self.ious_sum["overall"].append(iou_sum)
                self.ious_sum[gt].append(iou_sum)
                if batch["answer"][i] != 0:
                    self.ious_sum_nonzero["overall"].append(iou_sum)
                    self.ious_sum_nonzero[gt].append(iou_sum)

                # try a third method
                iou_boxes = 0
                iogt_boxes = 0
                ioo_boxes = 0
                for bbox, score in candidate_bbox:
                    iou_box, _, _, ioo_box, iogt_box = compute_iou(
                        batch["gt_bboxes"][i], [bbox]
                    )
                    iou_boxes += iou_box * score
                    ioo_boxes += ioo_box * score
                    iogt_boxes += iogt_box * score

                if "pred" in net_out:
                    pred = net_out["pred"][i].item()
                elif "counter-pred" in net_out:
                    pred = net_out["counter-pred"][i].item()
                    # print("counter-pred", pred)

                iou_boxes_norm = iou_boxes / pred
                ioo_boxes_norm = ioo_boxes / pred
                iogt_boxes_norm = iogt_boxes / pred

                # average precision
                # Predicted bounding boxes : numpy array [n, 4]
                # Predicted classes: numpy array [n]
                # Predicted confidences: numpy array [n]
                # Ground truth bounding boxes:numpy array [m, 4]
                # Ground truth classes: numpy array [m]
                # pred_bb1 = coords
                pred_bb = coords.cpu().numpy()
                pred_cls = np.zeros((len(pred_bb)))
                pred_conf = scores.view((scores.shape[0],)).cpu().numpy()
                gt_bb = np.array(batch["gt_bboxes"][i])
                gt_cls = np.zeros(len(gt_bb))
                if len(gt_bb) > 0:
                    for t in THRESHOLDS_mAP:
                        # breakpoint()
                        try:
                            self.mean_ap[t].evaluate(
                                pred_bb, pred_cls, pred_conf, gt_bb, gt_cls
                            )
                        except IndexError:
                            traceback.print_exc()
                            breakpoint()

                self.all_ious.append(
                    [
                        batch["question_id"][i],
                        batch["name"][i],
                        gt,
                        batch["gt_bboxes"][i],
                        pred,
                        round(pred),
                        candidate_bbox,
                        iou,
                        iou_sum,
                        ioo,
                        ioo_sum,
                        iogt,
                        iogt_sum,
                        iou_boxes,
                        iou_boxes_norm,
                        ioo_boxes,
                        ioo_boxes_norm,
                        iogt_boxes,
                        iogt_boxes_norm,
                    ]
                )

            iou = np.mean(ious)
            out["iou"] = iou
        return out


def compute_iou(gt_boxes, answer_boxes):
    """
    gt_boxes: [[x1, y1, x2, y2]]
    answer_boxes: [[x1, y1, x2, y2]]
    """
    gt_boxes_box = []
    for b in gt_boxes:
        gt_boxes_box.append(box(*b))
    union_boxes_gt = unary_union(gt_boxes_box)

    answer_boxes_box = []
    for b in answer_boxes:
        x, y, w, h = b
        answer_boxes_box.append(box(*b))
    union_answer_boxes = unary_union(answer_boxes_box)

    # intersection over union
    union = union_boxes_gt.union(union_answer_boxes)
    inter = union_boxes_gt.intersection(union_answer_boxes)

    if union.area == 0:
        iou = 1.0
    else:
        iou = inter.area / union.area

    if union_answer_boxes.area == 0:
        ioo = 1.0
    else:
        ioo = inter.area / union_answer_boxes.area

    if union_boxes_gt.area == 0:
        iogt = 1.0
    else:
        iogt = inter.area / union_boxes_gt.area

    return iou, inter.area, union.area, ioo, iogt

