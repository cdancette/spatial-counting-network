import os
import json
import re
import numpy as np
import io
import copy
import zipfile
import random
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.utils.data as data

from bootstrap import Logger
from bootstrap import Options
from bootstrap.datasets import transforms as bootstrap_tf


def tokenize_mcb(s):
    t_str = s.lower()
    for i in [
        r"\?",
        r"\!",
        r"\'",
        r"\"",
        r"\$",
        r"\:",
        r"\@",
        r"\(",
        r"\)",
        r"\,",
        r"\.",
        r"\;",
    ]:
        t_str = re.sub(i, "", t_str)
    for i in [r"\-", r"\/"]:
        t_str = re.sub(i, " ", t_str)
    q_list = re.sub(r"\?", "", t_str.lower()).split(" ")
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list


"""
Question looks like: 
 {'answer': 4,
  'data_source': 'imported_genome',
  'image': 'VG_100K_2/2410408.jpg',
  'image_id': 92410408,
  'issimple': False,
  'question': 'How many headlights does the black bus have?',
  'question_id': 30095774}
"""


class TallyQA(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_data,
        dir_coco,
        dir_vg,
        split,
        val_size=0.05,
        image_features="default",
        background_coco=None,
        background_vg=None,
        background=False,
        background_merge=2,
        proportion_opposite=0.0,  # not used
        train_selection=None,  # not used
        no_features=False,
        path_questions=None,
        sampling=None,
        shuffle=None,
        batch_size=None,
    ):
        super().__init__()
        self.dir_data = dir_data
        self.dir_coco = dir_coco
        self.dir_vg = dir_vg
        self.split = split
        self.image_features = image_features
        self.dir_coco_lvis = "data/vqa/coco/extract_rcnn/lvis"
        self.dir_vg_lvis = "data/vqa/vgenome/extract_rcnn/lvis"
        self.background_coco = background_coco
        self.background_vg = background_vg
        self.background = background
        self.background_merge = background_merge
        self.no_features = no_features
        self.val_size = val_size
        self.path_questions = path_questions  # to override path to questions (default dir_data/split.json)
        self.sampling = sampling
        self.shuffle = shuffle
        self.batch_size = batch_size

        if self.dir_coco.endswith(".zip"):
            self.zip_coco = None  # lazy loading zipfile.ZipFile(self.dir_coco)
        if self.dir_vg.endswith(".zip"):
            self.zip_vg = None  # lazy loading zipfile.ZipFile(self.dir_vg)
        if self.background_coco is not None and self.background_coco.endswith(".zip"):
            self.zip_bg_coco = None  # zipfile.ZipFile(self.background_coco)
        if self.background_vg is not None and self.background_vg.endswith(".zip"):
            self.zip_bg_vg = None  # lazy loading zipfile.ZipFile(self.background_vg)
        if self.dir_coco.endswith(".lmdb"):
            self.lmdb_coco = None

        if self.split not in ["train", "test"]:
            self.process_split()

        # path = os.path.join(self.dir_data, "processed", "questions.json")
        q_path = self.get_path_questions()  # train or test
        Logger()("Loading questions")
        with open(q_path) as f:
            self.questions = json.load(f)

        self.path_wid_to_word = os.path.join(
            self.dir_data, "processed", "wid_to_word.pth"
        )
        if os.path.exists(self.path_wid_to_word):
            self.wid_to_word = torch.load(self.path_wid_to_word)
        else:
            os.makedirs(os.path.join(self.dir_data, "processed"), exist_ok=True)
            word_list = self.get_token_list()
            self.wid_to_word = {wid + 1: word for wid, word in enumerate(word_list)}
            torch.save(self.wid_to_word, self.path_wid_to_word)

        self.word_to_wid = {word: wid for wid, word in self.wid_to_word.items()}

        self.aid_to_ans = [str(a) for a in list(range(16))]
        self.ans_to_aid = {ans: i for i, ans in enumerate(self.aid_to_ans)}
        self.collate_fn = bootstrap_tf.Compose(
            [
                bootstrap_tf.ListDictsToDictLists(),
                bootstrap_tf.PadTensors(
                    use_keys=[
                        "question",
                        "pooled_feat",
                        "cls_scores",
                        "rois",
                        "cls",
                        "cls_oh",
                        "norm_rois",
                    ]
                ),
                # bootstrap_tf.SortByKey(key='lengths'), # no need for the current implementation
                bootstrap_tf.StackTensors(),
            ]
        )

    def get_path_questions(self, split=None):
        if self.path_questions is not None:
            return self.path_questions
        split = self.split if split is None else split
        if split in ["train", "test"]:
            return os.path.join(self.dir_data, f"{split}.json")
        else:
            suffix = str(self.val_size)
            return os.path.join(self.dir_data, "processed", f"{split}-{suffix}.json")

    def process_split(self):
        train_noval_path = self.get_path_questions(split="train-noval")
        val_path = self.get_path_questions(split="val")
        train_path = os.path.join(self.dir_data, "train.json")
        if os.path.exists(train_noval_path) and os.path.exists(val_path):
            return
        else:
            with open(train_path) as f:
                train_questions = json.load(f)
            all_images = list(set([q["image"] for q in train_questions]))

            n_val = int(len(all_images) * self.val_size)
            random.shuffle(all_images)
            train_noval_i = set(all_images[n_val:])
            val_i = set(all_images[:n_val])
            train_noval = [q for q in train_questions if q["image"] in train_noval_i]
            val = [q for q in train_questions if q["image"] in val_i]
            with open(train_noval_path, "w") as f:
                json.dump(train_noval, f)
            with open(val_path, "w") as f:
                json.dump(val, f)

    def get_token_list(self):
        Logger()("Getting tokens list")

        tokens = set()
        for split in ["train", "test"]:
            q_path = os.path.join(self.dir_data, f"{split}.json")  # train or test
            with open(q_path) as f:
                questions = json.load(f)
            for q in tqdm(questions):
                tokens = tokens.union(tokenize_mcb(q["question"]))
        # breakpoint()
        tokens = list(tokens)
        tokens = tokens
        return tokens

    def __len__(self):
        return len(self.questions)

    def load_zipfile_item(self, zipfile, item):
        f = io.BytesIO(zipfile.read(item))
        return torch.load(f)

    def load_np_zipfile_item(self, zipfile, item):
        f = io.BytesIO(zipfile.read(item))
        return np.load(f, allow_pickle=True).item()

    def load_zipfiles(self):
        if self.dir_coco.endswith(".zip") and self.zip_coco is None:
            self.zip_coco = zipfile.ZipFile(self.dir_coco)
        if self.dir_vg.endswith(".zip") and self.zip_vg is None:
            self.zip_vg = zipfile.ZipFile(self.dir_vg)
        if (
            self.background_coco is not None
            and self.background_coco.endswith(".zip")
            and os.path.exists(self.background_coco)
            and self.zip_bg_coco is None
        ):
            self.zip_bg_coco = zipfile.ZipFile(self.background_coco)
        if (
            self.background_vg is not None
            and self.background_vg.endswith(".zip")
            and os.path.exists(self.background_vg)
            and self.zip_bg_vg is None
        ):
            self.zip_bg_vg = zipfile.ZipFile(self.background_vg)

    def add_image_features(self, question):
        if "COCO" in question["image"]:
            img_name = question["image"].split("/")[1] + ".pth"
            question["img_name"] = question["image"].split("/")[1]
            path = os.path.join(self.dir_coco, img_name)
            if self.dir_coco.endswith(".zip"):
                if self.zip_coco is None:
                    self.load_zipfiles()
                features = self.load_zipfile_item(self.zip_coco, img_name)
            else:
                features = torch.load(path)
            question["img_path"] = path
        elif "VG" in question["image"]:
            # VG_100K/4.jpg
            question["img_name"] = question["image"].split("/")[1]
            img_name = question["image"].split("/")[1]
            path = os.path.join(self.dir_vg, img_name)[:-4] + ".pth"
            if self.dir_vg.endswith(".zip"):
                if self.zip_vg is None:
                    self.load_zipfiles()
                features = self.load_zipfile_item(self.zip_vg, img_name[:-4] + ".pth")
            else:
                features = torch.load(path)
            question["img_path"] = path

        question["visual"] = features["pooled_feat"]
        question["coord"] = features["rois"]
        norm_rois = features.get("norm_rois", None)
        if norm_rois is None:
            rois = question["coord"]
            rois_min = 0
            rois_max, _ = rois.max(dim=0)
            question["norm_coord"] = question["coord"] / rois_max  # between 0 and 1
        else:
            question["norm_coord"] = norm_rois

        question["nb_regions"] = question["visual"].size(0)
        return question

    def add_resnet_image_features(self, question, key=""):
        if "COCO" in question["image"]:
            img_name = question["image"].split("/")[1] + ".pth"
            path = os.path.join(self.background_coco, img_name)
            question["img_name"] = question["image"].split("/")[1]
            if self.background_coco.endswith(".zip"):
                if self.zip_bg_coco is None:
                    self.load_zipfiles()
                features = self.load_zipfile_item(self.zip_bg_coco, img_name)
            else:
                features = torch.load(path)
        elif "VG" in question["image"]:
            # VG_100K/4.jpg
            img_name = question["image"].split("/")[1] + ".pth"
            question["img_name"] = question["image"].split("/")[1]
            path = os.path.join(self.background_vg, img_name) + ".pth"
            if self.background_vg.endswith(".zip"):
                if self.zip_bg_vg is None:
                    self.load_zipfiles()
                features = self.load_zipfile_item(self.zip_bg_vg, img_name)
            else:
                features = torch.load(path)

        question[f"{key}visual"] = features["pooled_feat"]
        question[f"{key}coord"] = features["rois"]
        norm_rois = features.get("norm_rois", None)
        if norm_rois is None:
            rois = question[f"{key}coord"]
            rois_min = 0
            rois_max, _ = rois.max(dim=0)
            question[f"{key}norm_coord"] = (
                question[f"{key}coord"] / rois_max
            )  # between 0 and 1
        else:
            question[f"{key}norm_coord"] = norm_rois

        if self.background_merge != 1:
            kernel = (self.background_merge, self.background_merge)
            stride = (self.background_merge, self.background_merge)
            v = features["pooled_feat"].transpose(0, 1)  # 2014, 14*14
            v = v.view(1, v.shape[0], 14, 14)  # 1, 2048, 14, 14
            v = torch.nn.functional.avg_pool2d(
                v, kernel_size=kernel, stride=stride
            )  # 1, dim, h, w
            new_size = v.shape[2]
            v = v.view(v.shape[1], v.shape[2] * v.shape[3]).transpose(0, 1)  # h*w, dim
            question[f"{key}visual"] = v

            # recalculer rois et norm_rois
            x0 = (
                torch.arange(0, new_size)[None, :]
                .repeat_interleave(new_size, dim=0)
                .float()
            )
            y0 = (
                torch.arange(0, new_size)[:, None]
                .repeat_interleave(new_size, dim=1)
                .float()
            )
            x1 = (
                torch.arange(1, new_size + 1)[None, :]
                .repeat_interleave(new_size, dim=0)
                .float()
            )
            y1 = (
                torch.arange(1, new_size + 1)[:, None]
                .repeat_interleave(new_size, dim=1)
                .float()
            )
            rois = torch.stack((x0, y0, x1, y1), dim=2).reshape(new_size * new_size, 4)
            norm_rois = rois / float(new_size)

            question[f"{key}coord"] = rois
            question[f"{key}norm_coord"] = norm_rois

        question[f"{key}nb_regions"] = question[f"{key}visual"].size(0)

        return question

    def add_lvis_image_feature(self, question):
        try:
            if "COCO" in question["image"]:
                img_name = question["image"].split("/")[1] + ".pth"
                path = os.path.join(self.dir_coco_lvis, img_name)
            elif "VG" in question["image"]:
                # VG_100K/4.jpg
                img_name = question["image"].split("/")[1]
                path = os.path.join(self.dir_vg_lvis, img_name) + ".pth"
            features = torch.load(path)
            question["visual_lvis"] = features["pooled_feat"]
            question["coord_lvis"] = features["rois"]
            norm_rois = features.get("norm_rois_lvis", None)
            if norm_rois is None:
                rois = question["coord_lvis"]
                rois_min = 0
                rois_max, _ = rois.max(dim=0)
                # between 0 and 1
                question["norm_coord_lvis"] = question["coord_lvis"] / rois_max
            else:
                question["norm_coord_lvis"] = norm_rois
            question["nb_regions_lvis"] = question["visual_lvis"].size(0)
        except FileNotFoundError:
            Logger()(
                f"Missing LVIS features for image {question['image']}",
                log_level=Logger.ERROR,
            )
            question["visual_lvis"] = torch.zeros(100, 1024)
            question["coord_lvis"] = torch.zeros(100, 4)
            question["norm_coord_lvis"] = torch.zeros(100, 4)
            question["nb_regions_lvis"] = 0
        return question

    def add_lxmert_image_features(self, question):
        img_name = question["image"].split("/")[1][:-4]

        if "COCO" in question["image"]:
            shelve = self.coco_shelves
        elif "VG" in question["image"]:
            shelve = self.vg_shelve
            # VG_100K/4.jpg
        features = shelve[img_name]

        obj_num = features["num_boxes"]
        feats = features["features"].copy()
        boxes = features["boxes"].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = features["img_h"], features["img_w"]
        original_boxes = boxes
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        question["visual"] = torch.tensor(feats)
        question["coord"] = torch.tensor(original_boxes)
        question["norm_coord"] = torch.tensor(boxes)
        question["nb_regions"] = obj_num
        question["original_boxes"] = torch.tensor(original_boxes)
        return question

    # @profile
    def add_vilbert_image_features(self, question):
        self.load_zipfiles()
        if "VG" in question["image"]:
            # print("getting a VG image from tallyqa")
            img_id = question["image"].split("/")[1][:-4]
            path = os.path.join(self.dir_vg, img_id + ".npy")
            item = self.load_np_zipfile_item(self.zip_vg, img_id + ".npy")

            features = item["features"].reshape(-1, 2048)
            boxes = item["bbox"].reshape(-1, 4)

            # image_id = item["image_id"]
            image_h = int(item["image_height"])
            image_w = int(item["image_width"])
            num_boxes = features.shape[0]
            g_feat = np.sum(features, axis=0) / num_boxes
            num_boxes = num_boxes + 1
            features = np.concatenate(
                [np.expand_dims(g_feat, axis=0), features], axis=0
            )

            image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            image_location[:, :4] = boxes
            image_location[:, 4] = (
                (image_location[:, 3] - image_location[:, 1])
                * (image_location[:, 2] - image_location[:, 0])
                / (float(image_w) * float(image_h))
            )
            image_location_ori = copy.deepcopy(image_location)
            image_location[:, 0] = image_location[:, 0] / float(image_w)
            image_location[:, 1] = image_location[:, 1] / float(image_h)
            image_location[:, 2] = image_location[:, 2] / float(image_w)
            image_location[:, 3] = image_location[:, 3] / float(image_h)

            g_location = np.array([0, 0, 1, 1, 1])
            image_location = np.concatenate(
                [np.expand_dims(g_location, axis=0), image_location], axis=0
            )

            g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
            image_location_ori = np.concatenate(
                [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
            )
        elif "COCO" in question["image"]:
            # print("Getting COCO image")
            if self.dir_coco.endswith(".lmdb") and self.lmdb_coco is None:
                self.lmdb_coco = ImageFeaturesH5Reader(self.dir_coco)
            # breakpoint()
            img_id = question["image_id"]  # = question["image"].split("/")[1]
            item = self.lmdb_coco.__getitem__(img_id)
            # breakpoint()
            features, num_boxes, image_location, image_location_ori = item

        max_region_num = 101
        max_seq_length = int(Options()["model.network.parameters.max_length"])
        mix_num_boxes = min(int(num_boxes), max_region_num)
        mix_boxes_pad = np.zeros((max_region_num, 5))
        ori_boxes_pad = np.zeros((max_region_num, 5))
        mix_features_pad = np.zeros((max_region_num, 2048))
        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < max_region_num:
            image_mask.append(0)
        mix_boxes_pad[:mix_num_boxes] = image_location[:mix_num_boxes]
        ori_boxes_pad[:mix_num_boxes] = image_location_ori[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()

        question["visual"] = features
        # question["num_boxes"] = torch.tensor(num_boxes).long()
        question["norm_coord"] = torch.tensor(mix_boxes_pad).float()
        question["image_mask"] = image_mask
        question["co_attention_mask"] = torch.zeros((max_region_num, max_seq_length))
        # question["image_location_ori"] = torch.tensor(image_location_ori)

        return question

    def question_to_wids(self, question):
        tokens = tokenize_mcb(question.strip())
        return torch.tensor([self.word_to_wid[token] for token in tokens])

    def __getitem__(self, index):
        # get image
        question = deepcopy(self.questions[index])
        question["index"] = index
        full_question = question["question"]
        # question tokens
        tokens = tokenize_mcb(full_question.strip())
        # token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        question["question_tokens"] = tokens
        question["raw_question"] = full_question
        # pad question

        question["question"] = torch.tensor(
            [self.word_to_wid[token] for token in tokens]
        )
        question["lengths"] = torch.tensor([len(tokens)])
        question["class_id"] = torch.tensor([question["answer"]])
        question["label"] = {str(question["answer"]): 1}  # for lxmert
        question["target"] = torch.zeros(len(self.aid_to_ans))
        question["target"][question["answer"]] = 1.0
        question["original_question"] = question["raw_question"]

        if self.no_features:
            return question
        try:
            if self.image_features == "default":
                question = self.add_image_features(question)
            elif self.image_features == "default+lvis":
                question = self.add_image_features(question)
                question = self.add_lvis_image_feature(question)
            elif self.image_features == "lxmert":
                question = self.add_lxmert_image_features(question)
            elif self.image_features == "resnet":
                question = self.add_resnet_image_features(question)
            elif self.image_features == "vilbert":
                question = self.add_vilbert_image_features(question)
            if self.background:
                question = self.add_resnet_image_features(question, key="background_")
                # breakpoint()

        except FileNotFoundError:
            Logger()(
                f"Missing image {question['image']} of question {question['question_id']}"
            )
            question["question"] = torch.tensor([0])
            question["class_id"] = torch.tensor([0])
            question["visual"] = torch.zeros(100, 2048)
            question["coord"] = torch.zeros(100, 4)
            question["norm_coord"] = torch.zeros(100, 4)
            question["nb_regions"] = 100

        return question

    def make_batch_loader(self, batch_size=None, shuffle=None):
        shuffle = self.shuffle if shuffle is None else shuffle
        if self.sampling is not None and shuffle:
            print(f"SAMPLING with {self.sampling}")
            num_answers = defaultdict(int)
            for q in self.questions:
                num_answers[q["answer"]] += 1
            if self.sampling == "uniform_answer":
                weights = [1 / num_answers[q["answer"]] for q in self.questions]
            sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(self))
            batch_loader = data.DataLoader(
                dataset=self,
                batch_size=Options()["dataset.batch_size"],
                sampler=sampler,
                shuffle=False,
                pin_memory=Options()["misc.cuda"],
                num_workers=Options()["dataset.nb_threads"],
                collate_fn=self.collate_fn,
            )
        else:
            batch_loader = data.DataLoader(
                dataset=self,
                batch_size=Options()["dataset.batch_size"],
                shuffle=self.shuffle if shuffle is None else shuffle,
                pin_memory=Options()["misc.cuda"],
                num_workers=Options()["dataset.nb_threads"],
                collate_fn=self.collate_fn,
                sampler=None,
            )
        return batch_loader


class TallyQAEvenOdd2(TallyQA):
    """
    Even numbers in training
    Odd numbers in validation
    """

    def __init__(
        self, *args, proportion_opposite=0.0, train_selection="even", **kwargs
    ):

        """
        proportion_opposite: the proportion of odd (resp even) questions in train (resp test) set.
            (if train = even, opposite if train = odd)
        """
        super().__init__(*args, **kwargs)
        self.proportion_opposite = proportion_opposite
        assert 0.0 <= proportion_opposite <= 1.0
        assert train_selection in ("even", "odd")
        self.train_selection = train_selection
        self.proportion_opposite = proportion_opposite

        print("************", self.get_path_questions_even_odd())
        if not os.path.exists(self.get_path_questions_even_odd()):
            self.process_even_odd()

        if "train" in self.split or "val" in self.split:
            mode = "train"
        else:
            mode = "test"

        if (mode, self.train_selection) in [("train", "even"), ("test", "odd")]:
            self.own_numbers = [0, 2, 4, 6, 8, 10, 12, 14]
            self.opposite_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
        elif (mode, self.train_selection) in [("train", "odd"), ("test", "even")]:
            self.own_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
            self.opposite_numbers = [0, 2, 4, 6, 8, 10, 12, 14]
        else:
            raise ValueError((mode, self.train_selection))

        with open(self.get_path_questions_even_odd()) as f:
            self.questions = json.load(f)
        Logger()(f"Number of questions in split {self.split}: {len(self.questions)}")

    def get_path_questions_even_odd(self):
        path_dataset = f"{self.split}-evenodd-train_{self.train_selection}-prop_{self.proportion_opposite}.json"
        path = os.path.join(self.dir_data, "processed", path_dataset)
        return path

    def process_even_odd(self):
        Logger()(
            f"Creating EvenOdd split for "
            f"train={self.train_selection} and proportion_opposite={self.proportion_opposite}"
        )
        # breakpoint()
        even_questions = []
        odd_questions = []
        for q in self.questions:
            if q["answer"] % 2 == 0:
                even_questions.append(q)
            else:
                odd_questions.append(q)

        # filter answers
        if "train" in self.split or "val" in self.split:
            mode = "train"
        else:
            mode = "test"

        if (mode, self.train_selection) in [("train", "even"), ("test", "odd")]:
            own_questions = even_questions
            opposite_questions = odd_questions
        elif (mode, self.train_selection) in [("train", "odd"), ("test", "even")]:
            own_questions = odd_questions
            opposite_questions = even_questions
        else:
            raise ValueError((mode, self.train_selection))

        opposite_questions_by_ans = defaultdict(list)
        for q in opposite_questions:
            if "issimple" in q:
                key = str(q["answer"]) + ("simple" if q["issimple"] else "complex")
                opposite_questions_by_ans[key].append(q)
            else:
                opposite_questions_by_ans[q["answer"]].append(q)

        # select proportion  opposite
        Logger()(
            f"Number of opposite questions by ans: "
            f"{ {a: len(opposite_questions_by_ans[a]) for a in opposite_questions_by_ans} }"
        )
        opposite_selected = []
        for ans in opposite_questions_by_ans:
            tot = len(opposite_questions_by_ans[ans])
            num_sample = int(tot * self.proportion_opposite)
            if num_sample == 0 and self.proportion_opposite != 0:
                num_sample = min(10, len(opposite_questions_by_ans[ans]))
            selected = random.sample(opposite_questions_by_ans[ans], num_sample)
            Logger()(f"Number of questions selected for ans {ans}: {len(selected)}")
            opposite_selected += selected

        Logger()(
            f"Dataset : Split: {self.split}, train_selection={self.train_selection}"
        )
        Logger()(f"Number of own questions: {len(own_questions)}")
        Logger()(f"Number of opposite questions: {len(opposite_selected)}")
        own_questions = own_questions + opposite_selected

        # save
        path = self.get_path_questions_even_odd()
        with open(path, "w") as f:
            json.dump(own_questions, f)


class TallyQAOddEven2(TallyQAEvenOdd2):
    def __init__(self, *args, **kwargs):
        kwargs["train_selection"] = "odd"
        super().__init__(*args, **kwargs)
