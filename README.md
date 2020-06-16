# Overcoming statistical shortcuts for open-ended visual counting


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Code for models

The code for our SCN model is located in the `counting/models/networks/attcount_mlb.py` file.

The code to create our ablated versions of TallyQA is located in `counting/datasets/tallyqa.py`

The loss we use is loacted in `counting/models/criterions/counting_regression.py`

## Data download

### TallyQA dataset

The datasets are available at https://github.com/manoja328/TallyQA_dataset

### Our MCD ablated TallyQA datasets

Download our ablated version by running the script `./counting/datasets/scripts/download_mcd.sh`

### Image features

Download images features by running the script `./counting/datasets/scripts/download_features.sh`

## Training

To train the model(s) in the paper, run this command:

### For SCN on tallyqa Odd-Even-90% strategy

```bash
python -m bootstrap.run \
-o counting/options/tallyqa-odd-even-val2-0.1/scn.yaml \
--exp.dir logs/tallyqa-odd-even-val2-0.1/scn
```

### For SCN on tallyqa Even-Odd-90% strategy


```bash
python -m bootstrap.run \
-o counting/options/tallyqa-even-odd-val2-0.1/scn.yaml \
--exp.dir logs/tallyqa-even-odd-val2-0.1/scn
```

### For SCN on original tallyqa dataset

```bash
python -m bootstrap.run \
-o counting/options/tallyqa/scn.yaml \
--exp.dir logs/tallyqa/scn
```

This will run training, evaluation and testing.

##  View results

```bash
python -m counting.compare-tally-val -d logs/tallyqa-odd-even-val2-0.1/scn logs/tallyqa-even-odd-val2-0.1/scn logs/tallyqa/scn
```

## COCO-Grounding dataset

Download the dataset by running the script `./counting/datasets/scripts/download_coco_ground.sh`

You can then run the evaluation on COCOGrounding by running the following command


```bash
python -m bootstrap.run \
-o path/to/trained/model/options.yaml \
--exp.resume "best_eval_epoch.accuracy_top1" \
--dataset.train_split \
--dataset.params.path_questions data/vqa/tallyqa/coco-ground.json \
--misc.logs_name "coco_ground_0.2" \
--model.metric.score_threshold_grounding 0.2 \
--dataset.name "counting.datasets.tallyqa.TallyQA"
```

To perform early stopping on the validation set (if you use an ablated MCD dataset), use `--exp.resume "best_validation_epoch.tally_acc.overall"` instead.

To check results, run the command

```bash
python -m counting.compare-grounding -d <exp-dir>
```


## Pretrained models

Download a pretrain model on 

- TallyQA Odd-Even-90\% : 

- TallyQA Even-Odd-90\% : 

- Original TallyQA : 
