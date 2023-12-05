from dataclasses import dataclass

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tqdm.auto import tqdm
from transformers import AutoTokenizer

KAGGLE = False

if KAGGLE:

    @dataclass
    class Config:
        data_external_path: str = "../input/daigt-one-place-all-data/concatenated.csv"
        data_train_path: str = "../input/llm-detect-ai-generated-text/train_essays.csv"
        data_test_path: str = "../input/llm-detect-ai-generated-text/test_essays.csv"

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


class Exp04:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # adjust tokenizer path to change base on wich environent we are
        # kaggle or local
        self.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
        self.tfidf = None
        self.clf = None
        self.predictions = None

    def train_tokenizer(self):
        """
        Train tokenizer on X_train
        """
        train_texts = self.X_train.select("text").to_numpy().flatten().tolist()
        self.tokenizer = self.tokenizer.train_new_from_iterator(train_texts, 52000)

    def tokenize(self):
        """
        Tokenize text
        """
        train_texts = self.X_train.select("text").to_numpy().flatten().tolist()
        test_texts = self.X_test.select("text").to_numpy().flatten().tolist()

        train_tokenized = []
        for text in tqdm(train_texts):
            text_tokenized = self.tokenizer.tokenize(text)
            train_tokenized.append(" ".join(text_tokenized))

        test_tokenized = []
        for text in tqdm(test_texts):
            text_tokenized = self.tokenizer.tokenize(text)
            test_tokenized.append(" ".join(text_tokenized))

        self.X_train = train_tokenized
        self.X_test = test_tokenized

    def train_tfidf(self):
        """
        Train tfidf vectorizer
        """
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(3, 5),
            lowercase=False,
            sublinear_tf=True,
            analyzer="word",
            strip_accents="unicode",
        )
        self.tfidf.fit(self.X_train)

    def vectorize(self):
        self.X_train = self.tfidf.transform(self.X_train)
        self.X_test = self.tfidf.transform(self.X_test)

    def exp04(self):
        """
        Train and evaluate model
        """
        clf = MultinomialNB()
        clf.fit(self.X_train, self.y_train)
        self.predictions = clf.predict_proba(self.X_test)[:, 1]
        self.clf = clf

        # calculate roc auc
        roc_auc = roc_auc_score(self.y_test, self.predictions)
        print(f"ROC AUC: {roc_auc}")

    def submission(self, real_test):
        """
        Generate submission file
        """
        real_test_text = real_test.select("text").to_numpy().flatten().tolist()

        real_test_tokenized = []
        for text in tqdm(real_test_text):
            text_tokenized = self.tokenizer.tokenize(text)
            real_test_tokenized.append(" ".join(text_tokenized))

        real_test_transformed = self.tfidf.transform(real_test_tokenized)
        predictions = self.clf.predict_proba(real_test_transformed)[:, 1]
        df = pl.DataFrame(
            {
                "id": real_test.select("id").to_numpy().flatten().tolist(),
                "generated": predictions,
            }
        )
        df.write_csv("submission_tfidf.csv")


if __name__ == "__main__":
    config = Config()
    data_preprocessing = DataPreprocessing(config)
    data_preprocessing.pre_processing()
    X_train, X_test, y_train, y_test = data_preprocessing.split_data()
    exp04 = Exp04(X_train, X_test, y_train, y_test)
    exp04.train_tokenizer()
    exp04.tokenize()
    exp04.train_tfidf()
    exp04.vectorize()
    exp04.exp04()

    if KAGGLE:
        exp04.submission(data_preprocessing.data_test)
