"""
Benchmark code created in collaboration:
Chris Deotte: https://www.kaggle.com/cdeotte
Udbhav Bamba: https://www.kaggle.com/ubamba98
Roman Solovyev: https://www.kaggle.com/zfturbo

Metric taken from CPMP: https://www.kaggle.com/cpmpml
https://www.kaggle.com/code/cpmpml/faster-metric-computation
"""

import numpy as np
import pandas as pd
import time
import math
from ensemble_boxes import *
import multiprocessing as mp
from functools import partial


def calc_overlap2(set_pred, set_gt):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter / len_pred
        return (overlap_1, overlap_2)
    except:  # at least one of the input is NaN
        return (0, 0)


def score_feedback_comp_micro2(pred_df, gt_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df['class'] == discourse_type,
                          ['id', 'predictionstring']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    overlaps = [calc_overlap2(*args) for args in zip(joined.predictionstring_pred,
                                                     joined.predictionstring_gt)]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) \
                              for overlap in overlaps]
    joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)
    tp_pred_ids = joined_tp \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'gt_id'])['pred_id'].first()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)

    matched_gt_ids = joined_tp['gt_id'].unique()
    unmatched_gt_ids = set(joined['gt_id'].unique()) - set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_score = score_feedback_comp_micro2(pred_df, gt_df, discourse_type)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


class_to_label = {
    'Claim': 0,
    'Evidence': 1,
    'Lead': 2,
    'Position': 3,
    'Concluding Statement': 4,
    'Counterclaim': 5,
    'Rebuttal': 6
}

label_to_class = {v: k for k, v in class_to_label.items()}


def preprocess_for_wbf(df_list):
    boxes_list = []
    scores_list = []
    labels_list = []

    max_box_value = -1
    for df in df_list:
        scores_list.append(df['scores'].values.tolist())
        labels_list.append(df['class'].map(class_to_label).values.tolist())
        predictionstring = df.predictionstring.str.split().values
        df_box_list = []
        for bb in predictionstring:
            b1 = float(bb[0])
            b2 = float(bb[-1])
            max_box_value = max(max_box_value, b1, b2)
            df_box_list.append([b1, b2])
        boxes_list.append(df_box_list)

    max_box_value += 1
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            boxes_list[i][j][0] /= max_box_value
            boxes_list[i][j][1] /= max_box_value

    return boxes_list, scores_list, labels_list, max_box_value


label_to_threshold = {
    0: 0.275,  # Claim
    1: 0.375,  # Evidence
    2: 0.325,  # Lead
    3: 0.325,  # Position
    4: 0.4,  # Concluding Statement
    5: 0.275,  # Counterclaim
    6: 0.275  # Rebuttal
}


def postprocess_for_wbf(idx, boxes_list, scores_list, labels_list, max_box_value):
    preds = []
    for box, score, label in zip(boxes_list, scores_list, labels_list):
        if score > label_to_threshold[label]:
            start = math.ceil(box[0] * max_box_value)
            end = int(box[1] * max_box_value)
            preds.append((idx, label_to_class[label], ' '.join([str(x) for x in range(start, end + 1)])))
    return preds


def generate_wbf_for_id(i, bench):
    df_list = []
    for j in range(len(bench)):
        df_list.append(bench[j][bench[j]['id'] == i])

    boxes_list, scores_list, labels_list, max_box_value = preprocess_for_wbf(df_list)
    nboxes_list, nscores_list, nlabels_list = weighted_boxes_fusion_1d(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=0.33,
        conf_type='avg'
    )
    ret = postprocess_for_wbf(i, nboxes_list, nscores_list, nlabels_list, max_box_value)
    return ret


if __name__ == '__main__':
    NUM_CORES = 3

    in_dir = './'
    benchmark_csv = [
        in_dir + 'lsg-large-ALL.csv',
        in_dir + 'longformer-lstm-ALL.csv',
        in_dir + 'deberta-jaccard-ALL.csv',
        in_dir + 'deberta-large-v3-ALL.csv',
        in_dir + 'deberta-xlarge-v2-ALL.csv',
        in_dir + 'bird-base-1024-ALL.csv',
        in_dir + 'deberta-large-ALL.csv',
        in_dir + 'deberta-xlarge-ALL.csv',
        in_dir + 'funnel-large-ALL.csv',
        in_dir + 'yoso-ALL.csv',
    ]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert (len(benchmark_csv) == len(weights))

    # Calculate score before WBF
    valid = pd.read_csv(in_dir + 'valid.csv')
    bench = []
    for csv_path in benchmark_csv:
        start_time = time.time()
        pred = pd.read_csv(csv_path).dropna()
        bench.append(pred)
        if 1:
            score = score_feedback_comp(pred, valid)
            print('CSV: {} Score: {:.4f} Time: {:.2f} sec'.format(csv_path, score, time.time() - start_time))
        else:
            print('CSV: {} Time: {:.2f} sec'.format(csv_path, time.time() - start_time))

    v_ids = bench[0]['id'].unique()
    v_class = bench[0]['class'].unique()

    start_time = time.time()
    if NUM_CORES == 0:
        list_of_list = []
        for i, id in enumerate(v_ids):
            print('Go id: {} [{}/{}]'.format(id, i, len(v_ids)))
            res = generate_wbf_for_id(id, bench=bench)
            list_of_list.append(res)
    else:
        with mp.Pool(NUM_CORES * 2) as p:
            list_of_list = p.map(partial(generate_wbf_for_id, bench=bench), v_ids)
    print('Time: {:.2f} sec. Len list of lists: {}'.format(time.time() - start_time, len(list_of_list)))

    preds = [x for sub_list in list_of_list for x in sub_list]
    sub = pd.DataFrame(preds)
    print("Final submission shape: {}".format(sub.shape))
    sub.columns = ["id", "class", "predictionstring"]

    f1s = []
    CLASSES = sub['class'].unique()
    print("Ensemble results:")
    for c in CLASSES:
        pred_df = sub.loc[sub['class'] == c].copy()
        gt_df = valid.loc[valid['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print("{}: {:.4f}".format(c, f1))
        f1s.append(f1)
    print('\nOverall: {:.4f}'.format(np.mean(f1s)))
