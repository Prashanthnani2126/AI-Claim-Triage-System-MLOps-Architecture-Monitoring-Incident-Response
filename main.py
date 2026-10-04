import random

import numpy as np
import pandas as pd

from Config import Config
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling import model_predict
from preprocess import de_duplication, get_input_data, noise_remover

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """Load the input data used by the starter prototype."""
    df = get_input_data()
    return df


def preprocess_data(df: pd.DataFrame):
    """Apply basic preprocessing used by the starter prototype."""
    df = de_duplication(df)
    df = noise_remover(df)
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)


if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype("U")
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype("U")

    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print("\n" + "=" * 80)
        print(f"Training starter model for group: {name}")
        print("=" * 80)

        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
