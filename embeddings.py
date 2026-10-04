import random

import numpy as np
import pandas as pd
from Config import Config

seed = 0
random.seed(seed)
np.random.seed(seed)


def get_tfidf_embd(df: pd.DataFrame):
    """Create TF-IDF features from the two text columns.

    This is a simple prototype implementation. Students should inspect whether
    this function is used in a way that is appropriate for model evaluation.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + " " + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X


def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)
