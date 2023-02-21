def get_samples_num_per_cls(dataset, imb_factor=None, num_meta=None, train_size=None):
    """
    > Given a dataset, a train size, and an imbalance factor, return the number of samples per class

    :param dataset: the dataset to use
    :param imb_factor: the imbalance factor. If it's None, then the dataset is balanced
    :param num_meta: number of meta-training samples
    :param train_size: the number of samples in the training set
    :return: The number of samples per class.
    """
    if dataset == "mimic":
        cls_num = 2
        sample_max = (train_size - num_meta) / cls_num

    if imb_factor is None:
        return [sample_max] * cls_num
    samples_num_per_cls = []
    for cls_idx in range(cls_num):
        num = sample_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        samples_num_per_cls.append(int(num))
    return samples_num_per_cls
