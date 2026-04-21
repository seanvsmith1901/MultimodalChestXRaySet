import pandas as pd


def build_master_df(projections_path, reports_path):
  proj = pd.read_csv(projections_path)
  rep = pd.read_csv(reports_path)

  proj = proj[proj["projection"] == "Frontal"]
  df = proj.merge(rep[["uid", "findings", "impression", "Problems", "indication"]], on="uid", how="inner")

  # new label - 0 if normal, 1 if abnormal
  def label_from_text(row):
    # don't put everythign in there, just a little bit.
    text = str(row.get("Problems", "")).lower().strip()
    return 0 if text == "normal" else 1

  df["abnormal_label"] = df.apply(label_from_text, axis=1)

  # print(df["abnormal_label"].value_counts())
  # print(df["abnormal_label"].value_counts(normalize=True))


  return df