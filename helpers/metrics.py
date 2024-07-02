import numpy as np
import sklearn.metrics
import copy


def compute_deferral_metrics(data_test, sens=True):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels', 'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    results["human_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"], data_test["labels"]
    )

    results["coverage"] = 1 - np.mean(data_test["defers"])
    # get classifier accuracy when defers is 0
    if len(data_test["defers"] == 0) > 0:
        results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
            data_test["preds"][data_test["defers"] == 0],
            data_test["labels"][data_test["defers"] == 0],
        )
    else:
        results["classifier_nondeferred_acc"] = None
    # get human accuracy when defers is 1
    if len(data_test["defers"] == 1) > 0:
        results["human_deferred_acc"] = sklearn.metrics.accuracy_score(
            data_test["hum_preds"][data_test["defers"] == 1],
            data_test["labels"][data_test["defers"] == 1],
        )
    else:
        results["human_deferred_acc"] = None
    # get system accuracy
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (1 - data_test["defers"])
        + data_test["hum_preds"] * (data_test["defers"]),
        data_test["labels"],
    )
    # find FPR for sensititive_param=1 and sensitive_param=0
    # FPR = FP / (FP + TN)
    if sens:
        if len(data_test["sensitive_params"]==1) > 0:
            C1 = sklearn.metrics.confusion_matrix(
                data_test["labels"][data_test["sensitive_params"] == 1],
                data_test["preds"][data_test["sensitive_params"] == 1],
            )
        if len(data_test["sensitive_params"]==2) > 0:
            C2 = sklearn.metrics.confusion_matrix(
                data_test["labels"][data_test["sensitive_params"] == 2],
                data_test["preds"][data_test["sensitive_params"] == 2],
            )
        FPR1 = C1[0, 1] / (C1[0, 1] + C1[1, 0])
        FPR2 = C2[0, 1] / (C2[0, 1] + C2[1, 0])
        TPR1 = C1[1, 1] / (C1[1, 1] + C1[0, 0])
        TPR2 = C2[1, 1] / (C2[1, 1] + C2[0, 0])
        # Positive predictive proportion
        PP1 = (C1[1, 1]+C1[0, 1])/(C1[1, 1]+C1[0, 1]+C1[1, 0]+C1[0, 0])
        PP2 = (C2[1, 1]+C2[0, 1])/(C2[1, 1]+C2[0, 1]+C2[1, 0]+C2[0, 0])
        results["DP"] = PP1 - PP2
        results["FPRdiff"] = FPR1 - FPR2
        results["EqualOpp"] = TPR1 - TPR2
        results["EqualOdds"] = np.max([np.abs(FPR1 - FPR2),
                                       np.abs(TPR2 - TPR1)])


    return results


def compute_classification_metrics(data_test):
    """compute metrics for just classification

    Args:
        data_test (dict): dict data with fields 'labels',  'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data, also returns AUC for preds_proba
    """

    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    # check if preds and labels are binary
    if (
        len(np.unique(data_test["labels"])) == 2
        and len(np.unique(data_test["preds"])) == 2
    ):
        # get f1
        results["classifier_all_f1"] = sklearn.metrics.f1_score(
            data_test["preds"], data_test["labels"]
        )
        if "preds_proba" in data_test:
            results["auc"] = sklearn.metrics.roc_auc_score(
                data_test["labels"], data_test["preds_proba"]
            )
        else:
            results["auc"] = sklearn.metrics.roc_auc_score(
                data_test["labels"], data_test["preds"]
            )
    return results


def compute_coverage_v_acc_curve(data_test):
    """

    Args:
        data_test (dict): dict data with field   {'defers': defers_all, 'labels': truths_all, 'hum_preds': hum_preds_all, 'preds': predictions_all, 'rej_score': rej_score_all, 'class_probs': class_probs_all}

    Returns:
        data (list): compute_deferral_metrics(data_test_modified) on different coverage levels, first element of list is compute_deferral_metrics(data_test)
    """
    # get unique rejection scores
    rej_scores = np.unique(data_test["rej_score"])
    # sort by rejection score
    # get the 100 quantiles for rejection scores
    rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    # for each quantile, get the coverage and accuracy by getting a new deferral decision
    all_metrics = []
    all_metrics.append(compute_deferral_metrics(data_test))
    for q in rej_scores_quantiles:
        # get deferral decision
        defers = (data_test["rej_score"] > q).astype(int)
        copy_data = copy.deepcopy(data_test)
        copy_data["defers"] = defers
        # compute metrics
        metrics = compute_deferral_metrics(copy_data)
        all_metrics.append(metrics)
    return all_metrics
