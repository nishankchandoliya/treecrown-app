import math
import os

from pipeline.validation import GTBox, evaluate, load_ground_truth_csv


def test_perfect_matches():
    gt = [GTBox(0, 0, 10, 10), GTBox(20, 20, 30, 30)]
    preds = [(0, 0, 10, 10, 0.9), (20, 20, 30, 30, 0.8)]
    r = evaluate(gt, preds, iou_threshold=0.5)
    assert r.true_positives == 2
    assert r.false_positives == 0
    assert r.false_negatives == 0
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0
    assert math.isclose(r.mean_iou_matched, 1.0, rel_tol=1e-9)


def test_hand_computed_mixed_scenario():
    """5 reference trees. 4 predictions: 3 exactly match 3 different
    reference trees (perfect IoU), 1 is a false positive with no nearby
    reference box. 2 reference trees are missed entirely.
    Expected: TP=3, FP=1, FN=2 -> precision=3/4=0.75, recall=3/5=0.6,
    F1 = 2*0.75*0.6/(0.75+0.6) = 0.6667.
    """
    gt = [
        GTBox(0, 0, 10, 10),
        GTBox(20, 0, 30, 10),
        GTBox(40, 0, 50, 10),
        GTBox(60, 0, 70, 10),   # will be missed
        GTBox(80, 0, 90, 10),   # will be missed
    ]
    preds = [
        (0, 0, 10, 10, 0.95),     # exact match to gt[0]
        (20, 0, 30, 10, 0.9),     # exact match to gt[1]
        (40, 0, 50, 10, 0.85),    # exact match to gt[2]
        (500, 500, 510, 510, 0.99),  # false positive, nowhere near any GT box
    ]
    r = evaluate(gt, preds, iou_threshold=0.5)
    assert r.true_positives == 3
    assert r.false_positives == 1
    assert r.false_negatives == 2
    assert math.isclose(r.precision, 0.75, rel_tol=1e-9)
    assert math.isclose(r.recall, 0.6, rel_tol=1e-9)
    assert math.isclose(r.f1, 2 * 0.75 * 0.6 / (0.75 + 0.6), rel_tol=1e-9)


def test_no_predictions_at_all():
    gt = [GTBox(0, 0, 10, 10), GTBox(20, 20, 30, 30)]
    r = evaluate(gt, [], iou_threshold=0.5)
    assert r.true_positives == 0
    assert r.false_negatives == 2
    assert r.false_positives == 0
    assert r.precision is None  # 0/0 undefined, not fabricated as 0 or 1
    assert r.recall == 0.0
    assert r.mean_iou_matched is None


def test_no_ground_truth_all_predictions_are_false_positives():
    r = evaluate([], [(0, 0, 10, 10, 0.9)], iou_threshold=0.5)
    assert r.true_positives == 0
    assert r.false_positives == 1
    assert r.precision == 0.0
    assert r.recall is None  # 0/0 undefined


def test_higher_confidence_prediction_wins_when_two_compete_for_one_gt_box():
    gt = [GTBox(0, 0, 10, 10)]
    preds = [
        (0, 0, 10, 10, 0.4),   # perfect box but lower confidence
        (1, 1, 11, 11, 0.9),   # slightly offset, higher confidence, still IoU>0.5
    ]
    r = evaluate(gt, preds, iou_threshold=0.5)
    assert r.true_positives == 1
    assert r.false_positives == 1  # the lower-confidence duplicate becomes an FP
    matched = [m for m in r.matches if m.kind == "tp"][0]
    assert predsindex_confidence(preds, matched.pred_index) == 0.9


def predsindex_confidence(preds, idx):
    return preds[idx][4]


def test_iou_threshold_affects_matching():
    gt = [GTBox(0, 0, 10, 10)]
    loose_pred = [(2, 2, 12, 12, 0.8)]  # partial overlap
    r_strict = evaluate(gt, loose_pred, iou_threshold=0.8)
    r_loose = evaluate(gt, loose_pred, iou_threshold=0.2)
    assert r_strict.true_positives == 0
    assert r_loose.true_positives == 1


def test_load_real_bundled_annotation_file():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "example", "OSBS_029.csv")
    boxes = load_ground_truth_csv(path, image_filename_filter="OSBS_029.tif")
    assert len(boxes) == 61  # matches `wc -l` (62 lines incl. header) minus header
    assert all(b.xmax > b.xmin and b.ymax > b.ymin for b in boxes)
    assert boxes[0].xmin == 203 and boxes[0].ymin == 67


def test_load_ground_truth_csv_round_trip(tmp_path):
    path = str(tmp_path / "gt.csv")
    with open(path, "w") as f:
        f.write("image_path,xmin,ymin,xmax,ymax,label\n")
        f.write("img.tif,1,2,3,4,Tree\n")
        f.write("img.tif,5,6,7,8,Tree\n")
        f.write("other.tif,9,9,9,9,Tree\n")  # should be excluded by filter
    boxes = load_ground_truth_csv(path, image_filename_filter="img.tif")
    assert len(boxes) == 2
    assert boxes[1].xmax == 7
