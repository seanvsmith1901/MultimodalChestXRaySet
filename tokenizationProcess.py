# this is where we will be doign the NLP portion of htis project
# our goal is take in the raw text and classify it as normal or abnormal for pneumonia.
# simple as.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(text, nlp):
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop # drop "the", "is", "a", etc.
        and not token.is_punct # drop punctuation
        and not token.is_space # drop whitespace
    ]
    return " ". join(tokens)

def get_clean_stuff(df, nlp, mode="sparse"):
    max_features = -1 # should brick if this doesn't go off correctly.

    if mode == "sparse":
        max_features = 200
        df["text"] = (df["indication"].fillna(""))

    if mode == "medium":
        max_features = 1000
        df["text"] = (
            df["indication"].fillna("") + " " +
            df["impression"].fillna("")
        )

    if mode == "full":
        max_features = 500
        df["text"] = (
            df["indication"].fillna("") + " " +
            df["findings"].fillna("") + " " +
            df["impression"].fillna("")
            # do NOT put problems in here. bad idea.
        )

    df["text_clean"] = df["text"].apply(tokenize, nlp=nlp)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3), # trigrams
        min_df=2, # ignore terms apearing only once
    )

    X = vectorizer.fit_transform(df["text_clean"]).toarray()
    y = df["abnormal_label"].values

    # X is now our feature matrix , where each row is a paitent, and each column is a wordweight
    return X, y # give those back to our other guy.
    # now that we have this, we should be able to feed this into the MLP.