import pandas as pd

def build_master_df(projections_path, reports_path):
  proj = pd.read_csv(projections_path)
  rep = pd.read_csv(reports_path)

  # we only want frontal views for now, as multiple angles will cause problems early on.
  proj = proj[proj["projection"] == "Frontal"] # thats it, none of this side angle stuff.

  # now we need to join based on the UID
  # also make sure to perserve the indication
  df = proj.merge(rep[["uid", "findings", "impression", "Problems", "indication"]], on="uid", how="inner")

  # now lets grab some pneumonia keywords, for funzies.
  pneumonia_keywords = ["consolidation", "opacity", "opacities", "infiltrate", "pneumonia", "airspace", "alveolar"]

  def label_from_text(row):
    text = " ".join([
      str(row.get("findings", "")),
      str(row.get("impression", "")),
      str(row.get("Problems", "")),
    ]).lower() # to prevent character overflow.
    return int(any(kw in text for kw in pneumonia_keywords))

  df["pneumonia_label"] = df.apply(label_from_text, axis=1) # BY ROW BABY

  print("total samples ", len(df))
  print("postive (1) ", df["pneumonia_label"].sum())
  print("negative (0) ", (df["pneumonia_label"]==0).sum())

  return df