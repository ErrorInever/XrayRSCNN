def collate(batch):
    """
    By default, torch stacks the input image to from a tensor of size N*C*H*W,
    so every image in the batch must have the same height and width.
    """
    return tuple(zip(*batch))
