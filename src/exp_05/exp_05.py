from dataclasses import dataclass

import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm
from transformers import AutoTokenizer

KAGGLE = False

if KAGGLE:

    @dataclass
    class Config:
        data_external_path: str = "../input/daigt-one-place-all-data/concatenated.csv"
        data_train_path: str = "../input/llm-detect-ai-generated-text/train_essays.csv"
        data_test_path: str = "../input/llm-detect-ai-generated-text/test_essays.csv"
        tokenizer_path: str = "../input/tokenizer-gpt2/"

else:

    @dataclass
    class Config:
        data_external_path: str = "data/curated/external/concatenated.csv"
        data_train_path: str = "data/train_essays.csv"
        data_test_path: str = "data/test_essays.csv"
        tokenizer_path: str = "models/tokenizer"


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
            self.data_external.filter(
                (pl.col("generated") == 1) | (pl.col("generated") == 0)
            )
            .unique("text")
            .select(["id", "text", "generated"])
        )

        df_train = self.data_train.select(["id", "text", "generated"])

        self.data_train = pl.concat([df, df_train]).sample(
            fraction=1, shuffle=True, seed=42
        )

        return self.data_train.select(["text"]), self.data_train.select(["generated"])

    def split_data(self):
        """
        Split data into train and test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_train.drop("generated"),
            self.data_train.select("generated"),
            test_size=0.4,
            random_state=42,
            stratify=self.data_train.select("generated"),
        )
        return X_train, X_test, y_train, y_test


class Exp05:
    def __init__(self, data_train, y_train, config: Config):
        self.data_train = data_train
        self.y_train = y_train
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.tfidf = None
        self.clf = None
        self.predictions = None

    def train_tokenizer(self):
        """
        Train tokenizer on X_train
        """
        train_texts = self.data_train.to_numpy().flatten().tolist()
        self.tokenizer = self.tokenizer.train_new_from_iterator(train_texts, 32000)

    def tokenize(self):
        """
        Tokenize text
        """
        train_texts = self.data_train.to_numpy().flatten().tolist()

        train_tokenized = []
        for text in tqdm(train_texts):
            text_tokenized = self.tokenizer.tokenize(text)
            train_tokenized.append(" ".join(text_tokenized))

        self.data_train = train_tokenized

    def train_tfidf(self):
        """
        Train tfidf vectorizer
        """
        self.tfidf = TfidfVectorizer(
            ngram_range=(3, 5),
            stop_words="english",
            lowercase=False,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.tfidf.fit(self.data_train)

    def vectorize(self):
        self.data_train = self.tfidf.transform(self.data_train)

    def exp05_w_cv(self):
        """
        Train and evaluate model
        """
        clf = MultinomialNB(alpha=0.2)
        models = [
            (
                "SGD",
                SGDClassifier(
                    max_iter=8000, tol=1e-4, loss="modified_huber", n_jobs=10
                ),
            ),
            ("Logistic Regression", LogisticRegression()),
            # ("Gradient Boosting", HistGradientBoostingClassifier()),
            ("Multinomial Naive Bayes", clf),
            # ("Naive Bayes", clf),
        ]

        for name, model in models:
            scores = cross_val_score(
                model,
                self.data_train,
                self.y_train.to_numpy().flatten().tolist(),
                cv=10,
                scoring="roc_auc",
            )
            print(f"Cross validation scores for {name}: {scores}")
            print(f"Mean cross validation score for {name}: {scores.mean()}")

    def exp05_model(self):
        """
        Train and evaluate model
        """
        clf = LogisticRegression()
        clf.fit(self.data_train, self.y_train)
        self.clf = clf

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

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model
        """
        train_texts = X_test.to_numpy().flatten().tolist()

        test_tokenized = []
        for text in tqdm(train_texts):
            text_tokenized = self.tokenizer.tokenize(text)
            test_tokenized.append(" ".join(text_tokenized))

        X_test = test_tokenized
        X_test = self.tfidf.transform(X_test)
        predictions = self.clf.predict_proba(X_test)[:, 1]
        print(f"ROC AUC score: {roc_auc_score(y_test, predictions)}")


if __name__ == "__main__":
    config = Config()
    data_preprocessing = DataPreprocessing(config)
    data_train, y_train_full = data_preprocessing.pre_processing()
    X_train, X_test, y_train, y_test = data_preprocessing.split_data()

    exp05 = Exp05(data_train, y_train_full, config)
    exp05.train_tokenizer()
    exp05.tokenize()
    exp05.train_tfidf()
    exp05.vectorize()
    exp05.exp05_w_cv()
    # exp05.evaluate_model(X_test, y_test)

    if KAGGLE:
        exp05.submission(data_preprocessing.data_test)
