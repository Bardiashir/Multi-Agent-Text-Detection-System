from datasets import load_dataset
from config import RANDOM_STATE, SAMPLE_SIZE


def load_sample(n=SAMPLE_SIZE):

    ds = load_dataset("yaful/MAGE", split="test")
    df = ds.to_pandas()
    df = df.drop(columns=["src"])
    sample = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    return sample
