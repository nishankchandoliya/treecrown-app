from pipeline.dedup import box_iou, merge_duplicates
from pipeline.schema import Detection


def make_det(id, xmin, ymin, xmax, ymax, score, tile_id=0):
    return Detection(id=id, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score=score, tile_id=tile_id)


def test_iou_identical_boxes_is_one():
    a = make_det(0, 0, 0, 10, 10, 0.9)
    b = make_det(1, 0, 0, 10, 10, 0.5)
    assert box_iou(a, b) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = make_det(0, 0, 0, 10, 10, 0.9)
    b = make_det(1, 100, 100, 110, 110, 0.5)
    assert box_iou(a, b) == 0.0


def test_merge_duplicates_at_tile_border_keeps_higher_score():
    # Same real-world tree, detected slightly differently by two overlapping
    # tiles near their shared border - classic tile-seam duplicate.
    det_from_tile_a = make_det(0, 100, 100, 140, 140, score=0.72, tile_id=0)
    det_from_tile_b = make_det(1, 103, 98, 143, 138, score=0.91, tile_id=1)  # same tree, slightly shifted box
    merged = merge_duplicates([det_from_tile_a, det_from_tile_b], iou_threshold=0.4)
    assert len(merged) == 1
    assert merged[0].score == 0.91  # kept the higher-confidence one


def test_merge_keeps_genuinely_distinct_trees():
    tree_1 = make_det(0, 0, 0, 40, 40, 0.8, tile_id=0)
    tree_2 = make_det(1, 500, 500, 540, 540, 0.7, tile_id=1)  # far away, distinct tree
    merged = merge_duplicates([tree_1, tree_2], iou_threshold=0.4)
    assert len(merged) == 2


def test_merge_empty_input():
    assert merge_duplicates([]) == []


def test_merge_three_way_cluster_keeps_only_best():
    # three overlapping detections of the same crown from 3 tiles meeting at a corner
    a = make_det(0, 10, 10, 50, 50, 0.6)
    b = make_det(1, 12, 8, 52, 48, 0.95)
    c = make_det(2, 8, 12, 48, 52, 0.4)
    merged = merge_duplicates([a, b, c], iou_threshold=0.3)
    assert len(merged) == 1
    assert merged[0].score == 0.95
