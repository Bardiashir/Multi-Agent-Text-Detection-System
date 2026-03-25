from datasets import load_dataset
import pandas as pd


def load_sample(n=20):
    
    ds = load_dataset("yaful/MAGE" , split="test")
    df = ds.to_pandas()
    df["text"] = df["text"].str.replace("\n", " ")
    df = df.drop(columns=["src"])
#1 : human , 0 : AI
    sample = df.sample(n=n , random_state= 42).reset_index(drop=True)
    return sample
