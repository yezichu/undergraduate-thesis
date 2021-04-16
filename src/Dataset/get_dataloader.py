from torch.utils.data import Dataset, DataLoader


def get_dataloader(
    dataset: Dataset,
    data_path: str,
    phase: str,
    transform,
    batch_size: int = 1,
    num_workers: int = 4,
):
    dataset = dataset(data_path, phase, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader
