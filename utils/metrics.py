from collections import Counter

def calculate_metrics(gt, pred):
    """
    Calculate micro and macro precision, recall, and F1 for lists of ground truth and predictions.

    Args:
        gt (list of list): Ground truth values.
        pred (list of list): Predicted values.

    Returns:
        metrics (list of tuples): List of (TP, FP, FN) for each instance.
        micro_metrics (tuple): Precision, recall, F1 (micro-averaged).
        macro_metrics (tuple): Precision, recall, F1 (macro-averaged).
    """

    # Store TP, FP, FN for each example
    metrics = []
    for g, p in zip(gt, pred):
        g_counts = Counter(g)
        p_counts = Counter(p)

        # True Positives (TP): Correctly predicted labels (limited by ground truth count)
        tp = sum(min(p_counts[label], g_counts[label]) for label in g_counts if label in p_counts)

        # False Positives (FP): Extra predicted labels beyond correct matches
        fp = sum(p_counts[label] - min(p_counts[label], g_counts.get(label, 0)) for label in p_counts)

        # False Negatives (FN): Missing ground truth labels
        fn = sum(g_counts[label] - min(g_counts[label], p_counts.get(label, 0)) for label in g_counts)

        metrics.append((tp, fp, fn))

    # Micro-averaging
    total_tp = sum(tp for tp, _, _ in metrics)
    total_fp = sum(fp for _, fp, _ in metrics)
    total_fn = sum(fn for _, _, fn in metrics)

    micro_precision = round(total_tp / (total_tp + total_fp),2) if (total_tp + total_fp) > 0 else 0
    micro_recall = round(total_tp / (total_tp + total_fn),2) if (total_tp + total_fn) > 0 else 0
    micro_f1 = round(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall),2) if (micro_precision + micro_recall) > 0 else 0

    # Macro-averaging
    precisions = []
    recalls = []
    f1_scores = []

    for tp, fp, fn in metrics:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = round(sum(precisions) / len(precisions),2) if precisions else 0
    macro_recall = round(sum(recalls) / len(recalls),2) if recalls else 0
    macro_f1 = round(sum(f1_scores) / len(f1_scores),2) if f1_scores else 0

    return metrics, (micro_precision, micro_recall, micro_f1), (macro_precision, macro_recall, macro_f1)


from collections import Counter

def calculate_metrics_set_wise(gt, pred):
    metrics = []

    for g, p in zip(gt, pred):
        g_set = set(g)
        p_set = set(p)

        # True Positives (TP): Correctly predicted labels
        tp = len(g_set & p_set)  # Intersection of ground truth and prediction

        # False Positives (FP): Extra predicted labels not in ground truth
        fp = len(p_set - g_set)  # Elements in pred but not in ground truth

        # False Negatives (FN): Missing ground truth labels
        fn = len(g_set - p_set)  # Elements in ground truth but not in pred

        metrics.append((tp, fp, fn))

    # Micro-averaging
    total_tp = sum(tp for tp, _, _ in metrics)
    total_fp = sum(fp for _, fp, _ in metrics)
    total_fn = sum(fn for _, _, fn in metrics)

    micro_precision = round(total_tp / (total_tp + total_fp), 2) if (total_tp + total_fp) > 0 else 0
    micro_recall = round(total_tp / (total_tp + total_fn), 2) if (total_tp + total_fn) > 0 else 0
    micro_f1 = round(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall), 2) if (micro_precision + micro_recall) > 0 else 0

    # Macro-averaging
    precisions, recalls, f1_scores = [], [], []

    for tp, fp, fn in metrics:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = round(sum(precisions) / len(precisions), 2) if precisions else 0
    macro_recall = round(sum(recalls) / len(recalls), 2) if recalls else 0
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 2) if f1_scores else 0

    return metrics, (micro_precision, micro_recall, micro_f1), (macro_precision, macro_recall, macro_f1)
