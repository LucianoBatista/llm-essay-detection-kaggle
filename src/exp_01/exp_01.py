from enum import Enum, auto
import uuid
import polars as pl
from dataclasses import dataclass


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
    augmented_data_13_path: str = "data/external/persuade15_claude_instant1.csv"
    augmented_data_14_path: str = (
        "data/external/persuade_2.0_human_scores_demo_id_github.csv"
    )
    augmented_data_17_path: str = "data/external/train_drcat_01.csv"
    augmented_data_20_path: str = "data/external/train_drcat_04.csv"
    feedback_prize_3_path: str = "data/external/feedback-prize-en-language-learning.csv"


class KaggleRepos(Enum):
    LLM_GEN_ESSAYS = auto()
    ARGUGPT = auto()
    DAIGT_EXTERNAL_DATASET = auto()
    DAIGT_LLAMA_FALCON = auto()
    DAIGT_PROPER_TRAIN_DATASET = auto()
    HELLO_CLAUDE_ESSAYS = auto()
    USING_PALM = auto()
    PERSUADE_CORPUS = auto()
    FEEDBACK_PRIZE_3 = auto()


def concat_data(config: Config, kaggle_repo: KaggleRepos):
    # id, prompt_id, text, generated, model
    aug_1 = (
        pl.read_csv(config.augmented_data_1_path)
        .with_columns(
            model=pl.lit("gpt-3.5-turbo"),
            kaggle_repo=pl.lit(kaggle_repo.LLM_GEN_ESSAYS.value),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_2 = (
        pl.read_csv(config.augmented_data_2_path)
        .with_columns(
            model=pl.lit("gpt-4"),
            kaggle_repo=pl.lit(kaggle_repo.LLM_GEN_ESSAYS.value),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_3 = (
        pl.read_csv(config.augmented_data_3_path)
        .with_columns(
            generated=pl.lit(1),
            model=pl.lit("unknown"),
            kaggle_repo=pl.lit(kaggle_repo.ARGUGPT.value),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_4 = (
        pl.read_csv(config.augmented_data_4_path)
        .with_columns(
            prompt_id=pl.lit(-1),
            generated=pl.lit(1),
            model=pl.lit("gpt-3.5-turbo"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_EXTERNAL_DATASET.value),
        )
        .select(["id", "prompt_id", "source_text", "generated", "model", "kaggle_repo"])
        .rename({"source_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    ## id, prompt_id, text, generated, model
    aug_5 = pl.read_csv(config.augmented_data_5_path)
    unique_ids_5 = [str(uuid.uuid4()) for _ in range(len(aug_5))]
    aug_5 = (
        pl.read_csv(config.augmented_data_5_path)
        .with_columns(
            id=pl.Series(unique_ids_5),
            prompt_id=pl.lit(-1),
            generated=pl.lit(1),
            model=pl.lit("falcon-180b"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_LLAMA_FALCON.value),
        )
        .select(
            ["id", "prompt_id", "generated_text", "generated", "model", "kaggle_repo"]
        )
        .rename({"generated_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_6 = pl.read_csv(config.augmented_data_6_path)
    unique_ids_6 = [str(uuid.uuid4()) for _ in range(len(aug_6))]
    aug_6 = (
        pl.read_csv(config.augmented_data_6_path)
        .with_columns(
            id=pl.Series(unique_ids_6),
            prompt_id=pl.lit(-1),
            generated=pl.lit(1),
            model=pl.lit("llama-70b"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_LLAMA_FALCON.value),
        )
        .select(
            ["id", "prompt_id", "generated_text", "generated", "model", "kaggle_repo"]
        )
        .rename({"generated_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_7 = pl.read_csv(config.augmented_data_7_path)
    unique_ids_7 = [str(uuid.uuid4()) for _ in range(len(aug_7))]
    aug_7 = (
        pl.read_csv(config.augmented_data_7_path)
        .with_columns(
            id=pl.Series(unique_ids_7),
            prompt_id=pl.lit(-1),
            generated=pl.lit(1),
            model=pl.lit("llama-70b"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_LLAMA_FALCON.value),
        )
        .select(
            ["id", "prompt_id", "generated_text", "generated", "model", "kaggle_repo"]
        )
        .rename({"generated_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_8 = pl.read_csv(config.augmented_data_8_path)
    unique_ids_8 = [str(uuid.uuid4()) for _ in range(len(aug_8))]
    aug_8 = (
        pl.read_csv(config.augmented_data_8_path)
        .with_columns(
            id=pl.Series(unique_ids_8),
            prompt_id=pl.lit(-1),
            generated=pl.lit(1),
            model=pl.lit("llama-falcon"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_LLAMA_FALCON.value),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_9 = (
        pl.read_csv(config.augmented_data_9_path)
        .with_columns(
            model=pl.lit("palm-2"),
            kaggle_repo=pl.lit(kaggle_repo.USING_PALM.value),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_10 = pl.read_csv(config.augmented_data_13_path)
    unique_ids_10 = [str(uuid.uuid4()) for _ in range(len(aug_10))]

    aug_10 = (
        pl.read_csv(config.augmented_data_13_path)
        .with_columns(
            prompt_id=pl.col("prompt_id") + 2,
        )
        # all others persuade prompts needs to subtract 2
        .with_columns(
            prompt_id=pl.when(pl.col("prompt_id") == 4)
            .then(pl.lit(0))
            .when(pl.col("prompt_id") == 14)
            .then(pl.lit(1))
            .otherwise(pl.col("prompt_id"))
        )
        .with_columns(
            id=pl.Series(unique_ids_10),
            generated=pl.lit(1),
            model=pl.lit("claude"),
            kaggle_repo=pl.lit(kaggle_repo.HELLO_CLAUDE_ESSAYS.value),
        )
        .rename({"essay_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_11 = pl.read_csv(config.augmented_data_14_path)
    unique_ids_11 = [str(uuid.uuid4()) for _ in range(len(aug_11))]

    # this text was written by a human
    aug_11 = (
        pl.read_csv(config.augmented_data_14_path)
        .with_columns(
            prompt_id=pl.when(pl.col("prompt_name") == "Car-free cities")
            .then(pl.lit(0))
            .when(pl.col("prompt_name") == "Does the electoral college work?")
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
        )
        .with_columns(
            id=pl.Series(unique_ids_11),
            generated=pl.lit(0),
            model=pl.lit("human"),
            kaggle_repo=pl.lit(kaggle_repo.PERSUADE_CORPUS.value),
        )
        .select(["id", "prompt_id", "full_text", "generated", "model", "kaggle_repo"])
        .rename({"full_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_12 = pl.read_csv(config.augmented_data_17_path).filter(
        pl.col("source") == "llammistral7binstruct"
    )
    unique_ids_12 = [str(uuid.uuid4()) for _ in range(len(aug_12))]

    # this text was written by a human
    aug_12 = (
        pl.read_csv(config.augmented_data_17_path)
        .filter(pl.col("source") == "llammistral7binstruct")
        .with_columns(
            id=pl.Series(unique_ids_12),
            generated=pl.lit(1),
            model=pl.lit("mistral"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_PROPER_TRAIN_DATASET.value),
            prompt_id=pl.lit(-1),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_13 = pl.read_csv(config.augmented_data_20_path).filter(
        pl.col("source") == "mistral7binstruct_v2"
    )
    unique_ids_13 = [str(uuid.uuid4()) for _ in range(len(aug_13))]

    # this text was written by a human
    aug_13 = (
        pl.read_csv(config.augmented_data_20_path)
        .filter(pl.col("source") == "mistral7binstruct_v2")
        .with_columns(
            id=pl.Series(unique_ids_13),
            generated=pl.lit(1),
            model=pl.lit("mistral"),
            kaggle_repo=pl.lit(kaggle_repo.DAIGT_PROPER_TRAIN_DATASET.value),
            prompt_id=pl.lit(-1),
        )
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    # id, prompt_id, text, generated, model
    aug_14 = pl.read_csv(config.feedback_prize_3_path)
    unique_ids_14 = [str(uuid.uuid4()) for _ in range(len(aug_14))]

    # this text was written by a human
    aug_14 = (
        pl.read_csv(config.feedback_prize_3_path)
        .with_columns(
            id=pl.Series(unique_ids_14),
            generated=pl.lit(0),
            model=pl.lit("human"),
            kaggle_repo=pl.lit(kaggle_repo.FEEDBACK_PRIZE_3.value),
            prompt_id=pl.lit(-1),
        )
        .rename({"full_text": "text"})
        .select(
            pl.col("id").cast(pl.Utf8),
            pl.col("prompt_id").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
            pl.col("generated").cast(pl.Int8),
            pl.col("model").cast(pl.Utf8),
            pl.col("kaggle_repo").cast(pl.Int32),
        )
    )

    concatenated = pl.concat(
        [
            aug_1,
            aug_2,
            aug_3,
            aug_4,
            aug_5,
            aug_6,
            aug_7,
            aug_8,
            aug_9,
            aug_10,
            aug_11,
            aug_12,
            aug_13,
            aug_14,
        ]
    )

    concatenated.write_csv("data/curated/external/concatenated.csv")


if __name__ == "__main__":
    config = Config()
    concat_data(config, KaggleRepos)
