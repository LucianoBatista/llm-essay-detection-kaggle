import polars as pl
from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = "data/train_essays.csv"
    data_kaggle_path: str = "data/train_essays.csv"


def main(config: Config):
    df = pl.read_csv(config.data_path)

    # target distribution
    print(df.group_by("generated").agg(pl.count()))
    print(df.shape)


if __name__ == "__main__":
    config = Config()
    main(config)
