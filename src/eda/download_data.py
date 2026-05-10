import os
from typing import Union

from huggingface_hub import login

from src import HFLogin, ROOT_PATH
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset

DATASET_PATH = ROOT_PATH/"data/Falah_skin_cancer_dataset.parquet"
DatasetType = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]

def download_dataset() -> DatasetType:
    login(HFLogin)
    DATASET_NAME = "Falah/skin-cancer"
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        verification_mode="no_checks"
    )
    dataset.to_parquet(DATASET_PATH)
    return dataset

def load_dataset_parquet() -> DatasetType:
    if os.path.exists(DATASET_PATH):
        return load_dataset(path=str(DATASET_PATH))

        #Pentru mai mult control
        dataset:Dataset = load_dataset(path=str(DATASET_PATH))
        return dataset

    return download_dataset()



