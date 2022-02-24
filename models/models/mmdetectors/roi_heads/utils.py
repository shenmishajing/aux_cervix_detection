import torch


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 5)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    bboxes = bboxes.detach()
    labels = labels.detach()
    return [bboxes[labels == i, :] for i in range(num_classes)]
