import torch
from typing import Union


def focal_loss_segmentation(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 3, beta: float = 4,
                            weights: Union[torch.Tensor, None] = None, verbose: bool = False) -> torch.Tensor:
    """
    Computes focal loss for heatmap prediction. Based on https://arxiv.org/pdf/1808.01244.pdf (see page 5)

    :param pred: prediction tensor
    :param gt: ground thruth tensor
    :param alpha: alpha hyperparameter of loss
    :param beta: beta hyperparameter of loss for the negative weight (the higher the less impact have negative samples)
    :param weights: class weights tensor
    :param verbose:
    :return:
    """

    if weights is not None:
        weights = weights.to(pred.device)
        try:
            weights = weights.view(1, pred.shape[1], 1, 1)
        except RuntimeError:
            raise ValueError("Bitte sicherstelle, dass in der Config nur die class weights für Klassen angegeben sind, "
                             "auf die auch trainiert wird!")



    # convert to probabilities
    pred_sigmoid = torch.sigmoid(pred)  # enable during training

    # Finding where the ground truth is positive (i.e., = 1)
    pos_inds = gt.eq(1).float()

    # Finding where the ground truth is negative (i.e., < 1)
    neg_inds = gt.lt(1).float()

    # TODO: compare num of pos indices to num of neg indices and tackle the class imbalance!!
    # Initialize loss to 0
    loss = 0

    # Calculate the negative weights
    neg_weights = torch.pow(1 - gt, beta)

    # Compute positive loss
    # WICHTIG: Ich habe 1e-7 verwendet, weil ab 1e-8 und der Verwendung von Mixed Precision für floating point numbers
    # die 1nur 16bit verwenden, dies nicht mehr berechnet werden kann und einfach als 0 ausgegeben wird
    # Trade off zwischen genaugigkeit und effizienz sehe ich effizienz vor, weil wir mit 1e-7 trotzdem noch echt gut
    # approximieren können und den loss nicht wirklich verzerren

    try:
    #TODO: aktuell sind predictions für center points oft so bei 0.1 - 0.2 --> hier sollte das modell noch sicherer werden
        pos_loss = torch.pow(1 - pred_sigmoid, alpha) * torch.log(pred_sigmoid + 1e-12) * pos_inds
    except RuntimeError:
        raise ValueError("Bitte prüfen ob Model Scale passt in der config file!")

    # Compute negative loss
    neg_loss = neg_weights * torch.pow(pred_sigmoid, alpha) * torch.log(1 - pred_sigmoid + 1e-7) * neg_inds

    # Count the number of positive and negative samples
    num_pos = pos_inds.float().sum()

    if weights is not None:
        pos_loss = pos_loss * weights
        neg_loss = neg_loss * weights

    if verbose:
        for i in range(pos_loss.shape[1]):
            try:
                print(f"Positive loss label {i}: {-pos_loss[i].sum()}")
                print(f"Negative loss label {i}: {-neg_loss[i].sum()}")
            except IndexError:
                print(f"Negative loss label {i}: {-neg_loss[i].sum()}")

    if num_pos == 0:
        loss = -neg_loss.sum()
    else:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    if torch.isnan(loss):
        print("is nan")

    return loss