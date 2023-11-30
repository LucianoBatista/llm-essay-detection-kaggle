from enum import Enum, EnumType, auto
import polars as pl
from dataclasses import dataclass
from pandas import read_csv


@dataclass
class Config:
    data_path: str = "data/train_essays.csv"
    data_kaggle_path: str = "data/train_essays.csv"
    augmented_data_1_path: str = "data/external/ai_generated_train_essays.csv"
    augmented_data_2_path: str = "data/external/ai_generated_train_essays_gpt-4.csv"
    augmented_data_3_path: str = "data/external/argugpt.csv"
    augmented_data_4_path: str = "data/external/daigt_external_dataset.csv"
    augmented_data_5_path: str = "data/external/falcon_180b_v1.csv"
    augmented_data_6_path: str = "data/external/llama_70b_v1.csv"
    augmented_data_7_path: str = "data/external/llama_70b_v2.csv"
    augmented_data_8_path: str = "data/external/llama_falcon_v3.csv"
    augmented_data_9_path: str = "data/external/LLM_generated_essay_PaLM.csv"
    augmented_data_10_path: str = "data/external/machine-dev.csv"
    augmented_data_11_path: str = "data/external/machine-test.csv"
    augmented_data_12_path: str = "data/external/machine-train.csv"
    augmented_data_13_path: str = "data/external/persuade15_claude_instant1.csv"
    augmented_data_14_path: str = (
        "data/external/persuade_2.0_human_scores_demo_id_github.csv"
    )
    augmented_data_15_path: str = "data/external/persuade_corpus_1.0.csv"
    augmented_data_16_path: str = "data/external/sources.csv"
    augmented_data_17_path: str = "data/external/train_drcat_01.csv"
    augmented_data_18_path: str = "data/external/train_drcat_02.csv"
    augmented_data_19_path: str = "data/external/train_drcat_03.csv"
    augmented_data_20_path: str = "data/external/train_drcat_04.csv"
    augmented_data_21_path: str = "data/external/train_essays_7_prompts.csv"
    augmented_data_22_path: str = "data/external/train_essays_7_prompts_v2.csv"
    augmented_data_23_path: str = "data/external/train_essays_RDizzl3_seven_v1.csv"
    augmented_data_24_path: str = "data/external/train_essays_RDizzl3_seven_v2.csv"
    augmented_data_25_path: str = "data/external/train_v2_drcat_02.csv"


class KaggleRepos(Enum):
    LLM_GEN_ESSAYS = auto()
    ARGUGPT = auto()
    DAIGT_EXTERNAL_DATASET = auto()
    DAIGT_LLAMA_FALCON = auto()
    DAIGT_PROPER_TRAIN_DATASET = auto()
    DAIGT_V2_TRAIN_DATASET = auto()
    HELLO_CLAUDE_ESSAYS = auto()
    USING_PALM = auto()
    LLM_7_PROMPT_TRAINING_DATASET = auto()
    PERSUADE_CORPUS = auto()


def concat_data(config: Config, kaggle_repo: KaggleRepos):
    # id, prompt_id, text, generated, model
    aug_1 = pl.read_csv(config.augmented_data_1_path).with_columns(
        model=pl.lit("gpt-3.5-turbo"),
        kaggle_repo=pl.lit(kaggle_repo.LLM_GEN_ESSAYS.value),
    )
    print(aug_1.shape)
    print(aug_1.select("generated").unique("generated"))
    print(aug_1.select("prompt_id").unique("prompt_id"))
    print(aug_1.head(5))

    # id, prompt_id, text, generated, model
    aug_2 = pl.read_csv(config.augmented_data_2_path).with_columns(
        model=pl.lit("gpt-4"),
        kaggle_repo=pl.lit(kaggle_repo.LLM_GEN_ESSAYS.value),
    )
    print(aug_2.shape)
    print(aug_2.select("generated").unique("generated"))
    print(aug_2.select("prompt_id").unique("prompt_id"))
    print(aug_2.head(5))

    # id, prompt_id, text, generated, model
    aug_3 = (
        pl.read_csv(config.augmented_data_3_path)
        .with_columns(
            generated=pl.lit(1),
            model=pl.lit("unknown"),
            kaggle_repo=pl.lit(kaggle_repo.ARGUGPT.value),
        )
        .select(["id", "prompt_id", "text", "generated", "model", "kaggle_repo"])
    )
    print(aug_3.shape)
    print(aug_3.head())


def from_kaggle(config: Config):
    curated_dataset = read_csv(
        "../input/daigt-proper-train-dataset/train_drcat_01.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset1 = read_csv(
        "../input/daigt-proper-train-dataset/train_drcat_02.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset2 = read_csv(
        "../input/daigt-proper-train-dataset/train_drcat_03.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset3 = read_csv(
        "../input/daigt-proper-train-dataset/train_drcat_04.csv"
    )[["text", "label"]].reset_index(drop=True)

    curated_dataset4 = (
        read_csv("../input/argugpt/machine-train.csv")[["text"]]
        .reset_index(drop=True)
        .assign(label=1)
    )
    curated_dataset5 = (
        read_csv("../input/argugpt/machine-test.csv")[["text"]]
        .reset_index(drop=True)
        .assign(label=1)
    )

    curated_dataset6 = (
        read_csv("../input/llm-generated-essays/ai_generated_train_essays.csv")[
            ["text"]
        ]
        .reset_index(drop=True)
        .assign(label=1)
    )
    curated_dataset7 = (
        read_csv("../input/llm-generated-essays/ai_generated_train_essays_gpt-4.csv")[
            ["text"]
        ]
        .reset_index(drop=True)
        .assign(label=1)
    )

    curated_dataset9 = (
        read_csv("/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv")[
            ["text"]
        ]
        .reset_index(drop=True)
        .assign(label=1)
    )
    curated_dataset10 = (
        read_csv("/kaggle/input/daigt-data-llama-70b-and-falcon180b/llama_70b_v1.csv")[
            ["generated_text"]
        ]
        .rename(columns={"generated_text": "text"})
        .reset_index(drop=True)
        .assign(label=1)
    )

    curated_dataset11 = (
        read_csv(
            "/kaggle/input/daigt-data-llama-70b-and-falcon180b/falcon_180b_v1.csv"
        )[["generated_text"]]
        .rename(columns={"generated_text": "text"})
        .reset_index(drop=True)
        .assign(label=1)
    )

    curated_dataset12 = read_csv(
        "/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset13 = (
        read_csv(
            "/kaggle/input/hello-claude-1000-essays-from-anthropic/persuade15_claude_instant1.csv"
        )[["essay_text"]]
        .rename(columns={"essay_text": "text"})
        .reset_index(drop=True)
        .assign(label=1)
    )
    curated_dataset14 = read_csv(
        "/kaggle/input/llm-generated-essay-using-palm-from-google-gen-ai/LLM_generated_essay_PaLM.csv"
    )[["text", "generated"]].rename(columns={"generated": "label"})
    curated_dataset15 = read_csv(
        "/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v2.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset16 = read_csv(
        "/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset17 = read_csv(
        "/kaggle/input/llm-7-prompt-training-dataset/train_essays_7_prompts_v2.csv"
    )[["text", "label"]].reset_index(drop=True)
    curated_dataset18 = read_csv(
        "/kaggle/input/llm-7-prompt-training-dataset/train_essays_7_prompts.csv"
    )[["text", "label"]].reset_index(drop=True)


def main(config: Config):
    df = pl.read_csv(config.data_path)

    # target distribution
    print(df.group_by("generated").agg(pl.count()))
    print(df.shape)


if __name__ == "__main__":
    config = Config()
    concat_data(config, KaggleRepos)
