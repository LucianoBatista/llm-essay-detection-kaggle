from dataclasses import dataclass

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

KAGGLE = False

if KAGGLE:

    @dataclass
    class Config:
        data_external_path: str = "../input/external/concatenated.csv"
        data_train_path: str = "../input/train_essays.csv"
        data_test_path: str = "../input/test_essays.csv"

else:

    @dataclass
    class Config:
        data_external_path: str = "data/curated/external/concatenated.csv"
        data_train_path: str = "data/train_essays.csv"
        data_test_path: str = "data/test_essays.csv"


class DataPreprocessing:
    def __init__(self, config: Config):
        self.data_external = pl.read_csv(
            config.data_external_path, infer_schema_length=40000
        )
        self.data_train = pl.read_csv(config.data_train_path)
        self.data_test = pl.read_csv(config.data_test_path).select(["id", "text"])

    def pre_processing(self):
        """
        1. Remove duplicates from external data
        2. Select only generated text and target
        3. Concatenate external data with train data
        """

        df = (
            self.data_external.filter(pl.col("kaggle_repo") != 6)
            .filter((pl.col("generated") == 1) | (pl.col("generated") == 0))
            .unique("text")
            .filter(
                (pl.col("prompt_id") == "1")
                | (pl.col("prompt_id") == "0")
                | (pl.col("prompt_id") == "2")
                | (pl.col("prompt_id") == "3")
                | (pl.col("prompt_id") == "4")
                | (pl.col("prompt_id") == "5")
                | (pl.col("prompt_id") == "6")
                | (pl.col("prompt_id") == "7")
            )
            .select(["id", "text", "generated"])
        )

        df_train = self.data_train.select(["id", "text", "generated"])

        self.data_train = pl.concat([df, df_train]).sample(
            fraction=1, shuffle=True, seed=42
        )

    def split_data(self):
        """
        Split data into train and test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_train.drop("generated"),
            self.data_train.select("generated"),
            test_size=0.2,
            random_state=42,
            stratify=self.data_train.select("generated"),
        )
        return X_train, X_test, y_train, y_test


class Exp03:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pipeline = None

    def get_pipeline(self):
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=4500)),
                ("clf", MultinomialNB()),
            ]
        )
        return pipeline

    def experiment(self):
        pipeline = self.get_pipeline()
        pipeline.fit(
            self.X_train.select("text").to_numpy().flatten().tolist(), self.y_train
        )
        self.predictions = pipeline.predict(
            self.X_test.select("text").to_numpy().flatten().tolist()
        )
        self.pipeline = pipeline

    def calculate_metrics(self):
        print(
            classification_report(
                self.y_test.to_numpy().flatten().tolist(), self.predictions
            )
        )

    def generate_submission(self, real_test):
        predictions = self.pipeline.predict_proba(
            real_test.select("text").to_numpy().flatten().tolist()
        )[:, 1]
        df = pl.DataFrame(
            {
                "id": real_test.select("id").to_numpy().flatten().tolist(),
                "generated": predictions,
            }
        )
        df.write_csv("submission.csv")


if __name__ == "__main__":
    config = Config()
    data_preprocessing = DataPreprocessing(config)
    data_preprocessing.pre_processing()
    X_train, X_test, y_train, y_test = data_preprocessing.split_data()

    print(X_train)

    exp03 = Exp03(X_train, X_test, y_train, y_test)
    exp03.experiment()
    exp03.calculate_metrics()
    exp03.generate_submission(data_preprocessing.data_test)
