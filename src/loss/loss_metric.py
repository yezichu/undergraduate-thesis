from __future__ import print_function, division
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
import torch
import torch.nn as nn
from medpy import metric
import warnings

warnings.simplefilter("ignore")


def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    #predictions =probabilities.float()
    assert (predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    # predictions =probabilities.float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def sensitivity_metric(probabilities: torch.Tensor,
                       truth: torch.Tensor,
                       treshold: float = 0.5,
                       eps: float = 1e-9) -> np.ndarray:

    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    # predictions =probabilities.float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (truth_ * prediction).sum()
        union = truth_.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(np.nan)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def specificity_metric(probabilities: torch.Tensor,
                       truth: torch.Tensor,
                       treshold: float = 0.5,
                       eps: float = 1e-9) -> np.ndarray:

    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    # predictions =probabilities.float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        tp = l_and(prediction, truth_).sum()
        tn = l_and(l_not(prediction), l_not(truth_)).sum()
        fp = l_and(prediction, l_not(truth_)).sum()
        #fn = l_and(l_not(prediction, truth_)).sum()
        #fn = np.sum(l_and(l_not(prediction, truth_)))
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(tn / (tn + fp))
        else:
            scores.append(tn / (tn + fp))
    return np.mean(scores)


def hausdorff_95(probabilities: torch.Tensor,
                 truth: torch.Tensor,
                 treshold: float = 0.5,
                 eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    # predictions =probabilities.float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(np.nan)
        else:
            scores.append(metric.hd95(prediction.numpy(), truth_.numpy()))
    return np.mean(scores)


def accuracy_metric(probabilities: torch.Tensor,
                    truth: torch.Tensor,
                    treshold: float = 0.5,
                    eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    # predictions =probabilities.float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        prediction = prediction.view(-1).data.cpu().numpy()
        truth_ = truth_.view(-1).data.cpu().numpy()
        (prediction == truth_).sum()
        scores.append((prediction == truth_).sum() / len(prediction))

    return np.mean(scores)


class Meter:
    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []
        self.sens_scores: list = []
        self.spec_scores: list = []
        self.accu_scores: list = []
        # self.haus_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        sens = sensitivity_metric(probs, targets, self.threshold)
        spec = specificity_metric(probs, targets, self.threshold)
        accu = accuracy_metric(probs, targets, self.threshold)
        # haus = hausdorff_95(probs, targets, self.threshold)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.sens_scores.append(sens)
        self.spec_scores.append(spec)
        self.accu_scores.append(accu)
        # self.haus_scores.append(haus)

    def get_metrics(self) -> np.ndarray:
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        sens = np.mean(self.sens_scores)
        spec = np.mean(self.spec_scores)
        accu = np.mean(self.accu_scores)
        # haus = np.mean(self.haus_scores)
        return dice, iou, sens, spec, accu


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert (probability.shape == targets.shape)

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        #self.dice = DiceLoss()
        self.contour = active_contour_loss()

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert (logits1.shape == targets.shape)
        assert (logits2.shape == targets.shape)
        #dice_loss = self.dice(logits, targets)
        bce_loss1 = self.bce(logits1, targets)
        bce_loss2 = self.bce(logits2, targets)
        bce_loss = bce_loss1 + bce_loss2
        contour_loss1 = self.contour(logits1, targets)
        contour_loss2 = self.contour(logits2, targets)
        contour_loss = contour_loss1 + contour_loss2

        return bce_loss + contour_loss


class active_contour_loss(nn.Module):
    """
    Active Contour Loss
    based on minpooling & maxpooling
    """
    def __init__(self, miu=1.0):
        super(active_contour_loss, self).__init__()

        self.miu = miu

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        min_pool_x = nn.functional.max_pool3d(logits * -1,
                                              (3, 3, 3), 1, 1) * -1
        contour = torch.relu(
            nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)

        # length
        length = torch.mean(torch.abs(contour))

        # region
        targets = targets.float()
        c_in = torch.ones_like(logits)
        c_out = torch.zeros_like(logits)
        region_in = torch.abs(torch.mean(logits * ((targets - c_in)**2)))
        region_out = torch.abs(
            torch.mean((1 - logits) * ((targets - c_out)**2)))
        region = self.miu * region_in + region_out

        return region + length
