from __future__ import print_function, division
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
from medpy import metric
import warnings

warnings.simplefilter("ignore")


def dice_coef_metric_per_classes(
        probabilities: np.ndarray,
        truth: np.ndarray,
        eps: float = 1e-9,
        classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """Calculate the dice for per classes.

    Args:
        probabilities (torch.Tensor): Predicted image.
        truth (torch.Tensor): The real image.
        eps (float, optional): Prevent divisor 0. Defaults to 1e-9.
        classes (list, optional): [description]. Defaults to ['WT', 'TC', 'ET'].

    Returns:
        np.ndarray: Return dice score.
    """
    scores = {key: list() for key in classes}
    num_classes = probabilities.shape[0]
    predictions = probabilities
    assert (predictions.shape == truth.shape)

    for class_ in range(num_classes):
        prediction = predictions[class_]
        truth_ = truth[class_]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores[classes[class_]].append(1.0)
        else:
            scores[classes[class_]].append((intersection + eps) / union)

    return scores


def jaccard_coef_metric_per_classes(
        probabilities: np.ndarray,
        truth: np.ndarray,
        eps: float = 1e-9,
        classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """Calculate the jaccard for per classes.

    Args:
        probabilities (torch.Tensor): Predicted image.
        truth (torch.Tensor): The real image.
        eps (float, optional): Prevent divisor 0. Defaults to 1e-9.
        classes (list, optional): [description]. Defaults to ['WT', 'TC', 'ET'].

    Returns:
        np.ndarray: Return jaccard score.
    """
    scores = {key: list() for key in classes}
    num_classes = probabilities.shape[0]
    predictions = probabilities
    assert (predictions.shape == truth.shape)

    for class_ in range(num_classes):
        prediction = predictions[class_]
        truth_ = truth[class_]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores[classes[class_]].append(1.0)
        else:
            scores[classes[class_]].append((intersection + eps) / union)

    return scores


def sensitivity_metric_per_classes(
        probabilities: np.ndarray,
        truth: np.ndarray,
        eps: float = 1e-9,
        classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """Calculate the sensitivity for per classes.

    Args:
        probabilities (torch.Tensor): Predicted image.
        truth (torch.Tensor): The real image.
        eps (float, optional): Prevent divisor 0. Defaults to 1e-9.
        classes (list, optional): [description]. Defaults to ['WT', 'TC', 'ET'].

    Returns:
        np.ndarray: Return sensitivity score.
    """
    scores = {key: list() for key in classes}
    num_classes = probabilities.shape[0]
    predictions = probabilities
    assert (predictions.shape == truth.shape)

    for class_ in range(num_classes):
        prediction = predictions[class_]
        truth_ = truth[class_]
        intersection = (truth_ * prediction).sum()
        union = truth_.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores[classes[class_]].append(np.nan)
        else:
            scores[classes[class_]].append((intersection + eps) / union)

    return scores


def specificity_metric_per_classes(
        probabilities: np.ndarray,
        truth: np.ndarray,
        eps: float = 1e-9,
        classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """Calculate the specificity for per classes.

    Args:
        probabilities (torch.Tensor): Predicted image.
        truth (torch.Tensor): The real image.
        eps (float, optional): Prevent divisor 0. Defaults to 1e-9.
        classes (list, optional): [description]. Defaults to ['WT', 'TC', 'ET'].

    Returns:
        np.ndarray: Return specificity score.
    """
    scores = {key: list() for key in classes}
    num_classes = probabilities.shape[0]
    predictions = probabilities
    assert (predictions.shape == truth.shape)

    for class_ in range(num_classes):
        prediction = predictions[class_]
        truth_ = truth[class_]
        tp = l_and(prediction, truth_).sum()
        tn = l_and(l_not(prediction), l_not(truth_)).sum()
        fp = l_and(prediction, l_not(truth_)).sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores[classes[class_]].append(tn / (tn + fp))
        else:
            scores[classes[class_]].append(tn / (tn + fp))

    return scores


def hausdorff_95_per_classes(probabilities: np.ndarray,
                             truth: np.ndarray,
                             eps: float = 1e-9,
                             classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """Calculate the hausdorff_95 for per classes.

    Args:
        probabilities (torch.Tensor): Predicted image.
        truth (torch.Tensor): The real image.
        eps (float, optional): Prevent divisor 0. Defaults to 1e-9.
        classes (list, optional): [description]. Defaults to ['WT', 'TC', 'ET'].

    Returns:
        np.ndarray: Return hausdorff_95 score.
    """
    scores = {key: list() for key in classes}
    num_classes = probabilities.shape[0]
    predictions = probabilities
    assert (predictions.shape == truth.shape)

    for class_ in range(num_classes):
        prediction = predictions[class_]
        truth_ = truth[class_]
        if truth_.sum() == 0 or prediction.sum() == 0:
            scores[classes[class_]].append(np.nan)
        else:
            scores[classes[class_]].append(metric.hd95(prediction, truth_))

    return scores
