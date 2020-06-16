import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import seaborn as sns
from collections import defaultdict
import json
from statistics import mean

from bootstrap.lib import utils
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import run
from tqdm import tqdm, tqdm_notebook
import matplotlib.patches as patches


def load_predictions(path_json):
    with open(path_json) as f:
        val_json = json.load(f)
    val_json = {item["question_id"]: item["answer"] for item in val_json}
    return val_json


def reset_options_instance():
    Options._Options__instance = None
    print("reset 2")
    Options.__instance = None
    Logger._Loger_instance = None
    Logger.perf_memory = {}
    sys.argv = [sys.argv[0]]  # reset command line args


def get_engine(
    path_experiment,
    weights="best_eval_epoch.accuracy_top1",
    load_original_annotations=True,
    options={},
):
    reset_options_instance()
    path_yaml = os.path.join(path_experiment, "options.yaml")
    opt = Options(path_yaml)
    opt["exp.resume"] = weights  # or "last"
    opt["exp.dir"] = path_experiment
    opt["misc.logs_name"] = "notebook"
    for arg in options:
        opt[arg] = options[arg]
    engine = run(train_engine=False, eval_engine=False)
    engine.dataset["eval"].load_original_annotation = load_original_annotations
    engine.dataset["train"].load_original_annotation = load_original_annotations
    return engine


def get_item(engine, idx=None, qid=None, split="eval"):
    if idx is None:
        idx = engine.dataset[split].qid_to_idx[qid]
    return engine.dataset[split][idx]


def get_item_to_batch(engine, idx=None, qid=None, prepare_batch=False):
    if idx is None:
        idx = engine.dataset["eval"].qid_to_idx[qid]
    item = engine.dataset["eval"][idx]
    batch = engine.dataset["eval"].collate_fn([item])
    engine.model.eval()
    if prepare_batch:
        batch = engine.model.prepare_batch(batch)
    return batch


def apply_model(engine, idx=None, qid=None, split="eval"):
    if idx is None:
        idx = engine.dataset[split].qid_to_idx[qid]
    item = engine.dataset[split][idx]
    batch = engine.dataset[split].collate_fn([item])
    engine.model.eval()
    batch = engine.model.prepare_batch(batch)
    with torch.no_grad():
        out = engine.model.network(batch)
    # out = engine.model.network.process_answers(out)
    # if 'logits_mm' in out:
    #     out = engine.model.network.process_answers(out, key='_mm')
    # if 'logits_q' in out:
    #     out = engine.model.network.process_answers(out, key='_q')
    # print("model answer:", out["answers_mm"][0])
    return out


def apply_item(engine, item, grad=False):
    torch.set_grad_enabled(grad)
    items = [item]
    batch = engine.dataset["eval"].collate_fn(items)
    batch = engine.model.cuda_tf()(batch)
    out = engine.model.network(batch)
    # out = engine.model.network.process_answers(out)
    torch.set_grad_enabled(True)
    return out


def get_image(engine, idx, dir_img="data/vqa/coco/raw"):
    item = engine.dataset["eval"][idx]
    split_name = "val2014" if "val2014" in item["image_name"] else "train2014"
    try:
        path_img = os.path.join(dir_img, split_name, item["image_name"])
    except:
        split_name = "train2014" if "val2014" in item["image_name"] else "val2014"
        if "val2014" in item["image_name"]:
            path_img = os.path.join(
                dir_img, split_name, item["image_name"].replace("val2014", "train2014")
            )
        else:
            path_img = os.path.join(
                dir_img, split_name, item["image_name"].replace("train2014", "val2014")
            )
    img = Image.open(path_img)
    return img


def get_img(image_name, dir_img="data/vqa/coco/raw"):
    split_name = "val2014" if "val2014" in image_name else "train2014"
    try:
        path_img = os.path.join(dir_img, split_name, image_name)
    except:
        split_name = "train2014" if "val2014" in image_name else "val2014"
        if "val2014" in image_name:
            path_img = os.path.join(
                dir_img, split_name, image_name.replace("val2014", "train2014")
            )
        else:
            path_img = os.path.join(
                dir_img, split_name, image_name.replace("train2014", "val2014")
            )
    img = Image.open(path_img)
    return img


def get_image_item(item, dir_img="data/vqa/coco/raw"):
    split_name = "val2014" if "val2014" in item["image_name"] else "train2014"
    try:
        path_img = os.path.join(dir_img, split_name, item["image_name"])
    except:
        split_name = "train2014" if "val2014" in item["image_name"] else "val2014"
        if "val2014" in item["image_name"]:
            path_img = os.path.join(
                dir_img, split_name, item["image_name"].replace("val2014", "train2014")
            )
        else:
            path_img = os.path.join(
                dir_img, split_name, item["image_name"].replace("train2014", "val2014")
            )
    img = Image.open(path_img)
    return img


def display_question(
    engine, idx=None, qid=None, dir_img="data/vqa/coco/raw", split="eval"
):
    if idx is None:
        idx = engine.dataset[split].qid_to_idx[qid]
    item = engine.dataset[split][idx]
    try:
        print("question_id:", item["question_id"])
        print("image_name:", item["original_question"]["image_name"])
        print("question:", item["original_question"]["question"])
        print("answer:", item["original_annotation"]["answer"])
    except:
        pass
    split_name = "val2014" if "val2014" in item["image_name"] else "train2014"
    try:
        path_img = os.path.join(dir_img, split_name, item["image_name"])
    except:
        split_name = "train2014" if "val2014" in item["image_name"] else "val2014"
        if "val2014" in item["image_name"]:
            path_img = os.path.join(
                dir_img, split_name, item["image_name"].replace("val2014", "train2014")
            )
        else:
            path_img = os.path.join(
                dir_img, split_name, item["image_name"].replace("train2014", "val2014")
            )
    img = Image.open(path_img)
    figsize = (4, 5)
    fig, ax = plt.subplots(figsize=figsize)
    # fig, ax = plt.subplots()
    ax.imshow(img)
    # ax.imshow(img_mask, alpha=0.6, cmap=cmap)
    ax.set_axis_off()
    plt.show()


def display_importance_map(
    engine,
    idx=None,
    qid=None,
    figsize=(18, 20),
    dir_img="data/vqa/coco/raw",
    logits="logits_mm",
    choice="max",
    split="eval",
):
    """
    Input:
        importance_map: 1D tensor of size n_regions
    """
    importance_map = get_importance_mapping(
        engine, qid=qid, idx=idx, logits=logits, choice=choice
    )

    if idx is None:
        idx = engine.dataset[split].qid_to_idx[qid]
    item = engine.dataset[split][idx]

    img = get_image(engine, idx, dir_img)

    cmap = plt.get_cmap("jet")

    width, height = img.size

    pix_mask = torch.zeros(height, width, 3)

    for i, roi in enumerate(item["coord"]):
        x, y, xx, yy = roi.tolist()
        w = round(xx - x)
        h = round(yy - y)
        x = round(x)
        y = round(y)
        cvalue = importance_map[i]  # mask[4][i] #/ 0.8
        # np.linalg.norm(buffer['x'], 0)
        pix_mask[y : y + h, x : x + w, :] += cvalue

    max_v = pix_mask.max()
    min_v = pix_mask.min()
    pix_mask = (pix_mask - min_v) / (max_v - min_v)
    pix_mask *= 255

    img_mask = Image.fromarray(np.uint8(pix_mask))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, alpha=0.8)
    ax.imshow(img_mask, alpha=0.6, cmap=cmap)
    ax.set_axis_off()

    plt.show()


def display_bounding_boxes_old(
    engine,
    idx,
    split="eval",
    weights=None,
    dir_img="data/vqa/coco/raw",
    figsize=(15, 15),
):
    img = get_image(engine, idx, dir_img)
    item = engine.dataset[split][idx]

    cmap = plt.get_cmap("jet")

    width, height = img.size
    pix_mask = torch.zeros(height, width, 3)

    for i, roi in enumerate(item["coord"]):
        x, y, xx, yy = roi.tolist()
        w = round(xx - x)
        h = round(yy - y)
        x = round(x)
        y = round(y)
        if weights is not None:
            cvalue = weights[i]  # mask[4][i] #/ 0.8
        else:
            cvalue = 1
        # np.linalg.norm(buffer['x'], 0)
        pix_mask[y : y + h, x : x + w, :] += cvalue

    max_v = pix_mask.max()
    min_v = pix_mask.min()
    pix_mask = (pix_mask - min_v) / (max_v - min_v)
    pix_mask *= 255
    img_mask = Image.fromarray(np.uint8(pix_mask))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, alpha=0.8)
    ax.imshow(img_mask, alpha=0.6, cmap=cmap)
    ax.set_axis_off()
    plt.show()


def display_bounding_boxes(
    img_path,
    boxes,
    box_normalized=False,
    img_name=None,
    weights=None,
    dir_img="data/vqa/coco/raw",
    figsize=(15, 15),
    ax=None,
):
    if img_path:
        img = Image.open(img_path)
    else:
        img = get_img(img_name, dir_img)
    img2 = img.copy()
    cmap = plt.get_cmap("jet")

    width, height = img.size
    pix_mask = torch.zeros(height, width, 3)

    draw = ImageDraw.Draw(img)

    for i, roi in enumerate(boxes):
        x, y, xx, yy = roi.tolist()
        if box_normalized:
            x = x * width
            xx = xx * width
            y = y * height
            yy = yy * height
        draw.rectangle([x, y, xx, yy], outline="red", width=1)
        w = round(xx - x)
        h = round(yy - y)
        x = round(x)
        y = round(y)
        if weights is not None:
            cvalue = weights[i]  # mask[4][i] #/ 0.8
        else:
            cvalue = 1
        pix_mask[y : y + h, x : x + w, :] += cvalue

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_axis_off()
    plt.show()




def display_attention(
    img_path,
    boxes,
    box_normalized=False,
    img_name=None,
    weights=None,
    dir_img="data/vqa/coco/raw",
    figsize=(15, 15),
    ax=None,
):
    if img_path:
        img = Image.open(img_path)
    else:
        img = get_img(img_name, dir_img)
    img2 = img.copy()
    cmap = plt.get_cmap("jet")

    width, height = img.size
    pix_mask = torch.zeros(height, width, 3)

    draw = ImageDraw.Draw(img)
    for i, roi in enumerate(boxes):
        x, y, xx, yy = roi.tolist()
        if box_normalized:
            x = x * width
            xx = xx * width
            y = y * height
            yy = yy * height
        # draw.rectangle([x,y,xx,yy], outline="red", width=1)
        w = round(xx - x)
        h = round(yy - y)
        x = round(x)
        y = round(y)
        if weights is not None:
            cvalue = weights[i]  # mask[4][i] #/ 0.8
        else:
            cvalue = 1
        pix_mask[y : y + h, x : x + w, :] += cvalue

    max_v = pix_mask.max()
    min_v = pix_mask.min()
    pix_mask = (pix_mask - min_v) / (max_v - min_v)
    pix_mask *= 255
    img_mask = Image.fromarray(np.uint8(pix_mask))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.imshow(img_mask, alpha=0.4, cmap=cmap)
    ax.set_axis_off()
    plt.show()
    # return img




def display_attention_red(
    img_path,
    boxes,
    box_normalized=False,
    img_name=None,
    weights=None,
    dir_img="data/vqa/coco/raw",
    figsize=(15, 15),
    ax=None,
):
    if img_path:
        img = Image.open(img_path)
    else:
        img = get_img(img_name, dir_img)

    width, height = img.size

    # ax1, ax2, ax3 = plt.figure()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    for i, roi in enumerate(boxes):
        x, y, xx, yy = roi.tolist()
        w = xx - x
        h = yy - y
        rect = patches.Rectangle(
            (x,y),w,h,
            linewidth=weights[i]*7,
            edgecolor='r',
            facecolor='none',
            label='lol')

        ax.add_patch(rect)
        rect = patches.Rectangle(
            (x,y),w,h,
            linewidth=0.5*weights[i],
            edgecolor='black',
            facecolor='none',
            label='lol')

        ax.add_patch(rect)
    # ax.imshow(img_mask, alpha=0.4, cmap=cmap)
    ax.set_axis_off()
    # plt.show()


def display_bounding_boxes_item(
    engine, item, weights=None, dir_img="data/vqa/coco/raw", figsize=(15, 15), ax=None,
):
    img = get_image_item(item, dir_img=dir_img)
    cmap = plt.get_cmap("jet")

    width, height = img.size
    pix_mask = torch.zeros(height, width, 3)

    for i, roi in enumerate(item["coord"]):
        x, y, xx, yy = roi.tolist()
        w = round(xx - x)
        h = round(yy - y)
        x = round(x)
        y = round(y)
        if weights is not None:
            cvalue = weights[i]  # mask[4][i] #/ 0.8
        else:
            cvalue = 1
        # np.linalg.norm(buffer['x'], 0)
        pix_mask[y : y + h, x : x + w, :] += cvalue

    max_v = pix_mask.max()
    min_v = pix_mask.min()
    pix_mask = (pix_mask - min_v) / (max_v - min_v)
    pix_mask *= 255
    img_mask = Image.fromarray(np.uint8(pix_mask))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, alpha=0.8)
    ax.imshow(img_mask, alpha=0.6, cmap=cmap)
    ax.set_axis_off()
    plt.show()


def load_model_state(engine, path):
    model_state = torch.load(path)
    engine.model.load_state_dict(model_state)


def load_epoch(
    engine,
    epoch,
    exp_dir="logs/vqacp2/murel_small_late_fusion_multiloss/best-save-weights/",
):
    path = os.path.join(exp_dir, f"ckpt_epoch_{epoch}_model.pth.tar")
    print(path)
    load_model_state(engine, path)


def load_last(
    engine, exp_dir="logs/vqacp2/murel_small_late_fusion_multiloss/best-save-weights"
):
    path = os.path.join(exp_dir, "ckpt_last_model.pth.tar")
    load_model_state(engine, path)


def get_idx(engine, idx=None, qid=None, split="eval"):
    if idx is None:
        idx = engine.dataset[split].qid_to_idx[qid]
    return idx


def grad_cam(engine, item, answer=None, logits="logits", negative=False, display=True):
    """
    answer: string or integer (answer_id). Default none is the highest scoring answer.
    """
    batch = engine.dataset["eval"].collate_fn([item])
    engine.model.eval()
    engine.model.network.set_buffer()
    engine.model.network.zero_grad()
    batch = engine.model.prepare_batch(batch)
    batch["visual"].requires_grad = True
    out = engine.model.network(batch)
    out = engine.model.network.process_answers(out)
    if answer is None:
        answer_id = torch.argmax(out[logits][0])
    elif isinstance(answer, str):
        answer_id = engine.model.network.ans_to_aid[answer]
    else:
        answer_id = answer
    if display:
        print(
            "Class used for importance mapping : %s"
            % engine.model.network.aid_to_ans[answer_id]
        )
    out[logits][0, answer_id].backward()

    # if question:
    #     region_grad = batch['visual'].grad[0].mean(dim=1).relu()
    #     word_embedding_grad = engine.model.network.buffer['q_emb'].grad[0].mean(dim=1).relu()
    #     question_grad = out['processed_question'].grad[0].max().item()
    #     visual_grad = batch['visual'].grad[0].max().item()
    #     return region_grad, word_embedding_grad, visual_grad, question_grad, visual_grad/question_grad*
    grad = batch["visual"].grad[0].mean(dim=1)
    if negative:
        grad = -grad
    return grad.relu()


def get_importance_mapping(
    engine, idx=None, qid=None, logits="logits_mm", choice="max", question=False
):
    """
    returns a tensor of size 36, with weights for every region
    Input:
        choice: either 'max' or an integer with the class_id, or a string
    """
    idx = get_idx(engine, idx, qid)
    item = engine.dataset["eval"][idx]
    batch = engine.dataset["eval"].collate_fn([item])
    engine.model.eval()
    engine.model.network.set_buffer()
    engine.model.network.zero_grad()
    batch = engine.model.prepare_batch(batch)
    batch["visual"].requires_grad = True
    out = engine.model.network(batch)
    out = engine.model.network.process_answers(out)
    if choice == "max":
        index = torch.argmax(out[logits][0])
    elif isinstance(choice, str):
        index = engine.model.network.ans_to_aid[choice]
    else:
        index = choice
    print(
        "Class used for importance mapping : %s"
        % engine.model.network.aid_to_ans[index]
    )
    out[logits][0, index].backward()

    if question:
        region_grad = batch["visual"].grad[0].mean(dim=1).relu()
        word_embedding_grad = (
            engine.model.network.buffer["q_emb"].grad[0].mean(dim=1).relu()
        )
        question_grad = out["processed_question"].grad[0].max().item()
        visual_grad = batch["visual"].grad[0].max().item()
        return (
            region_grad,
            word_embedding_grad,
            visual_grad,
            question_grad,
            visual_grad / question_grad,
        )
    return batch["visual"].grad[0].mean(dim=1).relu()


def apply_rubi(engine, idx=None, qid=None, topk=None, verbose=True, split="eval"):

    out = apply_model(engine, idx=idx, qid=qid, split=split)
    item = engine.dataset["eval"]
    if verbose:
        print("rubi mm: ", out["answers_mm"][0])
        print("rubi q: ", out["answers_q"][0])
        print("rubi fusion: ", out["answers"][0])
    if topk:
        for log in ("logits_mm", "logits_q"):
            logits = out[log].cpu().numpy()[0]
            index = np.argsort(logits)[::-1]
            topk_logits = logits[index[:k]]

        # engine.model.network.aid_to_ans()
    return out


def apply_baseline(engine, idx=None, qid=None):
    out = apply_model(engine, idx=idx, qid=qid)
    print("baseline: ", out["answers"][0])
    return out


def analyze_qid_baseline(engine, qid=None, idx=None):

    out = apply_baseline(engine, qid=qid, idx=idx)
    k = 10
    logits = out["logits"].cpu()[0]
    softmax = torch.softmax(logits, dim=0).numpy()
    logits = logits.numpy()
    indexes = np.argsort(logits)[::-1][:k]
    labels = [engine.model.network.aid_to_ans[id] for id in indexes]

    for name, log in (
        ("logits", logits),
        ("softmax", softmax),
    ):
        print(indexes)
        print(log[indexes])
        plt.figure(figsize=(12, 4))
        plt.title(name)
        sns.barplot(x=labels, y=log[indexes])
        plt.show()
    print("importance mapping")
    # im = get_importance_mapping(engine, qid=qid, idx=idx, logits="logits")
    display_importance_map(engine, qid=qid, idx=idx, logits="logits")


def compute_loss(engine, logits, true_label):
    prob = torch.softmax(torch.tensor(logits), dim=0)
    one_hot = torch.zeros(len(logits))
    aid = engine.model.network.ans_to_aid[true_label]
    # one_hot[aid] = 1
    return (-torch.log(1 - prob[aid])).item()


def analyze_qid_rubi(engine, qid=None, idx=None, true_label=None):
    display_question(engine, qid=qid, idx=idx)
    out = apply_rubi(engine, qid=qid, idx=idx)
    # get top 10 from mm and from fusion
    k = 10
    logits_mm = out["logits_mm"].cpu().numpy()[0]
    logits_fusion = out["logits"].cpu().numpy()[0]

    index_mm = np.argsort(logits_mm)[::-1][:k]
    index_fusion = np.argsort(logits_fusion)[::-1][:k]
    a = np.concatenate((index_mm, index_fusion), 0)
    indexes = np.unique(a, return_index=True)[1]
    indexes = [a[index] for index in sorted(indexes)]
    labels = [engine.model.network.aid_to_ans[id] for id in indexes]

    logits_fusion = out["logits"].cpu()[0]
    softmax_fusion = torch.softmax(logits_fusion, dim=0)
    logits_mm = out["logits_mm"].cpu()[0]
    softmax_mm = torch.softmax(logits_mm, dim=0)
    sigmoid_q = torch.sigmoid(out["logits_q_fusion"].cpu()[0])
    logits_q_final = out["logits_q"].cpu()[0]

    for name, log in (
        ("logits_mm", logits_mm),
        ("sigmoid_q_fusion", sigmoid_q),
        ("logits_fusion", logits_fusion),
        # ("softmax_mm", softmax_mm),
        # ("logits_q_final", logits_q_final),
        # ("softmax_fusion", softmax_fusion)
    ):
        plt.figure(figsize=(11, 3))
        plt.title(name)
        sns.barplot(x=labels, y=log[indexes])
        plt.show()

    if true_label:
        print("loss_mm", compute_loss(engine, logits_mm, true_label))
        print("loss_fusion", compute_loss(engine, logits_fusion, true_label))

    # print("importance mapping")
    # im = get_importance_mapping(engine, qid=qid)
    # display_importance_map(engine, im, qid=qid)
    print("===")


def plot_top_k(engine, logits, k=10):
    index = np.argsort(logits)[::-1][:k]
    labels = [engine.model.network.aid_to_ans[id] for id in index]
    plt.figure(figsize=(12, 4))
    sns.barplot(x=labels, y=logits[index])


def plot_logits_during_training(
    engine,
    qid,
    topk=3,
    exp_dir="logs/vqacp2/murel_small_late_fusion_multiloss/best-save-weights/",
    logits_name="logits_mm",
):

    # display_question(engine, qid=qid)

    result = defaultdict(list)
    keys = ["logits", "logits_mm", "logits_q_fusion"]

    for epoch in tqdm_notebook(range(0, 22)):
        path = os.path.join(exp_dir, f"ckpt_epoch_{epoch}_model.pth.tar")
        load_model_state(engine, path)
        out = apply_rubi(engine, qid=qid, verbose=False)
        for key in keys:
            vect = out[key].cpu()[0]
            result[key].append(vect.numpy())
            result[key + "_softmax"].append(torch.softmax(vect, dim=0).numpy())
            result[key + "_sigmoid"].append(torch.sigmoid(vect).numpy())

    n_epochs = len(result["logits"])

    ## get the top 3 indexes for the whole training
    all_index = np.array([]).astype("int")
    for logits in result["logits_mm"]:
        all_index = np.concatenate((all_index, np.argsort(logits)[::-1][:topk]))
    indexes = np.unique(all_index, return_index=True)[1]
    index = [all_index[i] for i in sorted(indexes)]
    labels = [engine.model.network.aid_to_ans[id] for id in index]

    fig = plt.figure(figsize=(12, 6))
    for i in range(len(labels)):
        data = [result["logits_mm"][epoch][index[i]] for epoch in range(n_epochs)]
        plt.plot(data, label=f"{labels[i]}_mm")
    plt.plot([0] * len(result["logits_mm"]), color="red", linestyle="--")
    plt.legend()
    plt.show()

    for k in range(len(index)):
        # plot for each class the three curves (mm, rubi, mask)
        data = [
            result["logits_mm_softmax"][epoch][index[k]] for epoch in range(n_epochs)
        ]
        plt.plot(data, label=f"{labels[k]}_mm")
        data = [result["logits_softmax"][epoch][index[k]] for epoch in range(n_epochs)]
        plt.plot(data, label=f"{labels[k]}_rubi")
        data = [
            result["logits_q_fusion_sigmoid"][epoch][index[k]]
            for epoch in range(n_epochs)
        ]
        plt.plot(data, label=f"{labels[k]}_q_mask")
        plt.title(labels[k])
        plt.legend()
        plt.show()

    return result


def plot_losses_during_training(
    engine,
    qid,
    topk=3,
    exp_dir="logs/vqacp2/murel_small_late_fusion_multiloss/best-save-weights/",
    logits_name="logits_mm",
    split="eval",
):

    # display_question(engine, qid=qid)

    item = get_item(engine, qid=qid, split=split)

    losses_original_list = []
    losses_rubi_list = []

    for epoch in tqdm_notebook(range(0, 22)):
        path = os.path.join(exp_dir, f"ckpt_epoch_{epoch}_model.pth.tar")
        load_model_state(engine, path)

        out = apply_rubi(engine, qid=qid, verbose=False, split=split)
        loss_rubi = F.cross_entropy(out["logits"].cpu(), item["class_id"])
        losses_rubi_list.append(loss_rubi)
        loss_orig = F.cross_entropy(out["logits_mm"].cpu(), item["class_id"])
        losses_original_list.append(loss_orig)
        # tqdm.write(f"original: {loss_orig}, rubi: {loss_rubi}")

    n_epochs = len(losses_original_list)

    ## get the top 3 indexes for the whole training
    # all_index = np.array([]).astype('int')
    # for logits in logits_list:
    #     all_index = np.concatenate((all_index, np.argsort(logits)[::-1][:topk]))
    # indexes = np.unique(all_index, return_index=True)[1]
    # index = [all_index[i] for i in sorted(indexes)]
    # labels =  [engine.model.network.aid_to_ans[id] for id in index]

    fig = plt.figure(figsize=(7, 3))
    plt.plot(losses_original_list, label="Original loss")
    plt.plot(losses_rubi_list, label="rubi loss")
    plt.legend()
    fig = plt.figure(figsize=(7, 3))
    plt.plot(
        [(rubi - orig) for (rubi, orig) in zip(losses_rubi_list, losses_original_list)],
        label="rubi - original",
    )
    plt.plot([0] * len(losses_rubi_list), color="red", linestyle="--")
    plt.legend()

    # for i in range(len(labels)):
    #     data = [logits_list[epoch][index[i]] for epoch in range(n_epochs)]
    #     plt.plot(data, label=labels[i])

    # plt.plot([0]*len(logits_list), color="red", linestyle="--")
    return losses_original_list, losses_rubi_list


def compute_mean(
    engine,
    qids,
    split="eval",
    exp_dir="logs/vqacp2/murel_small_late_fusion_multiloss/best-save-weights/",
):

    losses_rubi = []
    losses_original = []
    losses_qid = {"rubi": defaultdict(list), "original": defaultdict(list)}

    for epoch in tqdm_notebook(range(0, 22)):
        losses_rubi_e = []
        losses_original_e = []
        path = os.path.join(exp_dir, f"ckpt_epoch_{epoch}_model.pth.tar")
        load_model_state(engine, path)
        for qid in qids:
            if qid not in engine.dataset[split].qid_to_idx:
                continue
            item = get_item(engine, qid=qid, split=split)
            out = apply_rubi(engine, qid=qid, verbose=False, split=split)
            loss_rubi = F.cross_entropy(out["logits"].cpu(), item["class_id"]).item()
            losses_rubi_e.append(loss_rubi)
            loss_orig = F.cross_entropy(out["logits_mm"].cpu(), item["class_id"]).item()
            losses_original_e.append(loss_orig)
            losses_qid["rubi"][qid].append(loss_rubi)
            losses_qid["original"][qid].append(loss_orig)
        losses_rubi.append(losses_rubi_e)
        losses_original.append(losses_original_e)
    plt.plot([mean(x) for x in losses_rubi], label="rubi")
    plt.plot(
        [mean([mean(x) for x in losses_rubi])] * len(losses_rubi),
        label="rubi mean",
        linestyle="--",
    )
    plt.plot([mean(x) for x in losses_original], label="original")
    plt.plot(
        [mean([mean(x) for x in losses_original])] * len(losses_original),
        label="original mean",
        linestyle="--",
    )
    plt.legend()

    plt.figure()
    plt.plot(
        [mean(x) - mean(y) for x, y in zip(losses_rubi, losses_original)],
        label="rubi mean - original mean",
    )
    plt.plot(
        [mean([mean(x) - mean(y) for x, y in zip(losses_rubi, losses_original)])]
        * len(losses_rubi),
        linestyle="--",
        label="mean of differences",
    )
    plt.plot([0] * len(losses_rubi), color="red", linestyle="--")
    plt.legend()

    return losses_rubi, losses_original, losses_qid


from scipy.stats import spearmanr


def compute_vqa_hat_correlation(engine, question_id, attention):
    path_raw = "/net/pascal/cadene/data/block.bootstrap/coco/raw/val2014/"
    path_dataset = "/local/cadene/data/vqa-hat/vqahat_val"

    path_image = os.path.join(path_dataset, f"{question_id}_1.png")
    gt_attention_map = Image.open(path_image)  # (height, width)
    gt_attention_map_arr = np.array(gt_attention_map)

    rank_gt = np.array(gt_attention_map.resize((14, 14))).flatten()

    rank_net = attention  # 14*14 attention map
    correlation = spearmanr(rank_net, rank_gt)[0]

    pass
