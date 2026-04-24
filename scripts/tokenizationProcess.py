# this is where we will be doign the NLP portion of htis project
# our goal is take in the raw text and classify it as normal or abnormal for pneumonia.
# simple as.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
import medspacy

# Tokenizatino through Negex, a better NLP model for text recognition.
def build_nlp_pipeline():
    nlp = spacy.load("en_core_web_sm")

    ts = termset("en_clinical")
    nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

    return nlp



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

# negspacy flavor
def tokenize_with_negation(text, nlp):
    # instead of just returning tokens, prefix negated terms with NEG to TF-IDF treats them as distinc features.

    doc = nlp(str(text).lower())
    tokens = []

    for token in doc:
        # skip stopwords, punctuiation, whitespace
        if token.is_stop or token.is_punct or token.is_space:
            continue

        lemma = token.lemma_

        is_negated = False
        for ent in doc.ents:
            if token in ent and ent._.negex:
                is_negated = True
                break

        if is_negated:
            tokens.append(f"NEG_{lemma}")
        else:
            tokens.append(lemma)

    return " ". join(tokens)


# negex flavor
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

    # df["text_clean"] = df["text"].apply(tokenize, nlp=nlp)
    df["text_clean"] = df["text"].apply(tokenize_with_negation, nlp=nlp)

    print("here are the amx featuerts ", max_features)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3), # trigrams
        min_df=2, # ignore terms apearing only once
    )

    X = vectorizer.fit_transform(df["text_clean"]).toarray()
    y = df["pneumonia_label"].values

    # X is now our feature matrix , where each row is a paitent, and each column is a wordweight
    return X, y, df # give those back to our other guy.
    # now that we have this, we should be able to feed this into the MLP.


def has_pneumonia_using_pipeline(text, nlp, pneumonia_terms):
    if not isinstance(text, str) or not text.strip():
        return False # make sure it is a string we can parse

    doc = nlp(text.lower())

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        lemma = token.lemma_
        if any(term in lemma for term in pneumonia_terms):
            # check negation, same logic as in tokenize with negation
            is_negated = False
            for ent in doc.ents:
                if token in ent and ent._.negex:
                    is_negated = False
                    break

            if not is_negated:
                return True
            

    return False

