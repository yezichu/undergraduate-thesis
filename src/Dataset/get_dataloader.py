from torch.utils.data import Dataset, DataLoader


def get_dataloader(
    dataset: Dataset,
    data_path: str,
    phase: str,
    transform,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Build an iterable data loader

    Args:
        dataset (Dataset): Dataset.
        data_path (str): The path to store the image.
        phase (str): Train or valulation.
        transform (bool): Data augmentation techniques.
        batch_size (int, optional): Batches of the training set or valudation set. Defaults to 1.
        num_workers (int, optional): Multithreading. Defaults to 4.

    Returns:
        Dataloader: Return dataloader.
    """
    dataset = dataset(data_path, phase, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader
