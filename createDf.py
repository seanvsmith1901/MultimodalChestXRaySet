import pandas as pd


def build_master_df(projections_path, reports_path):
  proj = pd.read_csv(projections_path)
  rep = pd.read_csv(reports_path)

  proj = proj[proj["projection"] == "Frontal"]
  df = proj.merge(rep[["uid", "findings", "impression", "Problems", "indication"]], on="uid", how="inner")

  pneumonia_keywords = ["consolidation", "opacity", "opacities", "infiltrate",
                        "pneumonia", "airspace", "alveolar"]

  def label_from_text(row):
    # don't put everythign in there, just a little bit.
    text = str(row.get("Problems", "")).lower()
    return int(any(kw in text for kw in pneumonia_keywords))

  df["pneumonia_label"] = df.apply(label_from_text, axis=1)
  return df