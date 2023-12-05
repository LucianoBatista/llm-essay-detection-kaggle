from dataclasses import dataclass

import polars as pl
import torch
import torch.nn as nn
import torchdata.datapipes as dp
import torchtext
import torchtext.transforms as T
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchtext.functional import to_tensor
from tqdm.auto import tqdm

KAGGLE = False

if KAGGLE:

    @dataclass
    class Config:
        data_external_path: str = "../input/daigt-one-place-all-data/concatenated.csv"
        data_train_path: str = "../input/llm-detect-ai-generated-text/train_essays.csv"
        data_test_path: str = "../input/llm-detect-ai-generated-text/test_essays.csv"
        vocab_path: str = "../input/xlmr-base-pretrained-model/xlmr.vocab.pt"
        sentence_piece_tokenizer_path: str = (
            "../input/xlmr-base-pretrained-model/sentencepiece.bpe.model"
        )
        model_encoder_weights: str = (
            "../input/xlmr-base-pretrained-model/xlmr.base.encoder.pt"
        )

else:

    @dataclass
    class Config:
        data_external_path: str = "data/curated/external/concatenated.csv"
        data_train_path: str = "data/train_essays.csv"
        data_test_path: str = "data/test_essays.csv"
        vocab_path: str = "data/xlmr.vocab.pt"
        sentence_piece_tokenizer_path: str = "data/sentencepiece.bpe.model"
        model_encoder_weights: str = "data/xlmr.base.encoder.pt"


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


class Exp02:
    def __init__(
        self,
        data_train: pl.DataFrame,
        y_train: pl.DataFrame,
        data_test: pl.DataFrame,
        y_test: pl.DataFrame,
        config: Config = Config(),
    ):
        self.train = data_train
        self.y_train = y_train
        self.test = data_test
        self.y_test = y_test
        self.config = config
        self.head = self.get_head()
        self.model, self.transform_fn = self.get_model()
        self.datapipes = self.get_datapipe()
        self.pad_id = self.get_pad_id()

    def get_pad_id(self):
        padding_idx = self.transform_fn[1].vocab.lookup_indices(["<pad>"])[0]
        return padding_idx

    def get_transforms(self):
        vocab = torch.load(self.config.vocab_path)
        return T.Sequential(
            T.SentencePieceTokenizer(self.config.sentence_piece_tokenizer_path),
            T.VocabTransform(vocab),
            T.Truncate(256),
            T.AddToken(token=0, begin=True),
            T.AddToken(token=2, begin=False),
        )

    def get_model(self):
        xlmr_large = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_large.get_model(head=None, load_weights=False)
        model.load_state_dict(torch.load(self.config.model_encoder_weights))
        model.head = self.head
        transform_fn = self.get_transforms()
        return model, transform_fn

    def get_head(self):
        head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim=768)
        return head

    def apply_transform(self, row):
        text = row["text"]
        label = row["generated"]
        ids_ = row["id"]
        return (self.transform_fn(text), label, ids_)

    def tensor_batch(self, batch):
        tokens = batch["text"]
        labels = batch["generated"]
        ids = batch["id"]
        tokens_tensor = to_tensor(tokens, padding_value=self.pad_id)
        labels_tensor = torch.tensor(labels)
        return tokens_tensor, labels_tensor, ids

    def get_datapipe(self) -> dict:
        train = pl.concat([self.train, self.y_train], how="horizontal")
        test = pl.concat([self.test, self.y_test], how="horizontal")

        # train pipe
        datapipe_train = dp.iter.IterableWrapper(train.to_dicts())
        datapipe_train = datapipe_train.map(self.apply_transform)
        datapipe_train = datapipe_train.batch(16)
        datapipe_train = datapipe_train.rows2columnar(["text", "generated", "id"])
        datapipe_train = datapipe_train.map(self.tensor_batch)

        # test pipe
        datapipe_test = dp.iter.IterableWrapper(test.to_dicts())
        datapipe_test = datapipe_test.map(self.apply_transform)
        datapipe_test = datapipe_test.batch(32)
        datapipe_test = datapipe_test.rows2columnar(["text", "generated", "id"])
        datapipe_test = datapipe_test.map(self.tensor_batch)

        datapipe = {"train": datapipe_train, "test": datapipe_test}
        return datapipe

    def train_exp02(self):
        dataloaders = {}
        dataloaders["train"] = DataLoader(
            self.datapipes["train"], batch_size=None, shuffle=True
        )
        dataloaders["test"] = DataLoader(self.datapipes["test"], batch_size=None)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        epochs = 1
        epoch_losses = []
        epoch_val_losses = []
        device = "cuda"

        if not KAGGLE:
            writer = SummaryWriter("runs/script_experiment_1")

        self.model.to(device)

        for epoch in tqdm(range(epochs)):
            batch_losses = []
            for i, (tokens, labels, _) in enumerate(dataloaders["train"]):
                self.model.train()
                tokens = tokens.to(device)
                labels = labels.to(device)

                preds = self.model(tokens)
                loss = loss_fn(preds, labels)
                loss.backward()

                batch_losses.append(loss.item())
                if not KAGGLE:
                    writer.add_scalars(
                        main_tag="loss",
                        tag_scalar_dict={"training": loss.item()},
                        global_step=i,
                    )

                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch: {epoch} Train Loss: {sum(batch_losses) / len(batch_losses)}")
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
            val_losses = []

            with torch.inference_mode():
                for i, (tokens, labels, _) in enumerate(dataloaders["test"]):
                    self.model.eval()
                    tokens = tokens.to(device)
                    labels = labels.to(device)

                    preds = self.model(tokens)
                    loss = loss_fn(preds, labels)
                    val_losses.append(loss.item())

                    if not KAGGLE:
                        writer.add_scalars(
                            main_tag="loss",
                            tag_scalar_dict={"testing": loss.item()},
                            global_step=i,
                        )

                print(f"Epoch: {epoch} Val Loss: {sum(val_losses) / len(val_losses)}")
                epoch_val_losses.append(sum(val_losses) / len(val_losses))

    def predict(self, text: str, categories: list, proba: bool = False):
        """
        Predict of a single text
        """
        self.model.eval()
        tokens = self.transform_fn(text)
        tokens_tensor = to_tensor(tokens, padding_value=self.pad_id)
        tokens_to_pred = tokens_tensor.unsqueeze(0).to("cuda")
        preds = self.model(tokens_to_pred)

        # probabilities
        preds = torch.nn.functional.softmax(preds[0], dim=0)
        values, indices = torch.topk(preds, 1)

        # formated ouput
        output = {"label": categories[indices[0]], "value": values.item()}

        if proba:
            return values.item()

        return indices[0].item()

    def calculate_test_metrics(self, y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1


if __name__ == "__main__":
    config = Config()

    # data prep
    data_prep = DataPreprocessing(config)
    data_prep.pre_processing()
    X_train, X_test, y_train, y_test = data_prep.split_data()

    categories = ["human_generated", "ai_generated"]

    # modeling
    exp = Exp02(X_train, y_train, X_test, y_test)
    exp.train_exp02()
    texts = X_test.select("text").to_numpy().flatten().tolist()
    results = [exp.predict(text, categories) for text in texts]

    # calculating acc score on the test set
    if not KAGGLE:
        y_test_list = y_test.to_numpy().flatten().tolist()
        accuracy = accuracy_score(y_test_list, results)
        roc_auc = roc_auc_score(y_test_list, results)
        print(accuracy, roc_auc)
    else:
        private_test = data_prep.data_test
        # ids = private_test.select("id").to_numpy().flatten().tolist()
        texts = private_test.select("text").to_numpy().flatten().tolist()
        results = [exp.predict(text, categories) for text in texts]
        submission = private_test.with_columns(generated=pl.Series(results)).drop(
            "text"
        )
        submission.write_csv("submission.csv")
