import argparse, os, sys, pandas as pd
from tqdm import tqdm
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

# ----------------------------- CLI ------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="Input CSV with harassment rows")
ap.add_argument("--out_har", default="harassment_filtered.csv",
                help="Output CSV for harassment rows (label 1)")
ap.add_argument("--out_norm", default="normal_from_har.csv",
                help="Output CSV for non-bullying rows (label 0)")
args = ap.parse_args()

jar   = os.path.join("stanford-ner-2020-11-17/stanford-ner-4.2.0.jar")
model = os.path.join("stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz")

if not (os.path.isfile(jar) and os.path.isfile(model)):
    sys.exit("✖︎  stanford-ner.jar or model not found—check --ner_dir path.")

# ------------------------- Load dataset -------------------------------
df = pd.read_csv(args.csv)
print(f"Loaded {len(df):,} rows from {args.csv}")

# change these to your column names if different
text_col   = "comment_text"
toxic_cols = ["malignant","highly_malignant","rude","threat","abuse","loathe"]

if not all(c in df.columns for c in toxic_cols+[text_col]):
    sys.exit("✖︎  Expected toxicity columns not found—inspect CSV header.")

df["is_toxic"]     = df[toxic_cols].any(axis=1)
df["is_nonbully"]  = ~df["is_toxic"]

# ---------------------- Stanford NER setup ----------------------------
print("Initialising StanfordNERTagger …")
tagger = StanfordNERTagger(model, jar, encoding="utf-8")

def has_person_or_org(text:str) -> bool:
    tokens = word_tokenize(str(text))
    tags   = tagger.tag(tokens)
    return any(t[1] in ("PERSON","ORGANIZATION") for t in tags)

tqdm.pandas(desc="NER filter")
df["keep"] = df[text_col].progress_apply(has_person_or_org)

# ---------------------- Split & save ----------------------------------
harassment = df[df["is_toxic"] & df["keep"]][[text_col]].copy()
harassment["label"] = 1
harassment.to_csv(args.out_har, index=False)
print(f"Harassment rows saved → {args.out_har}  ({len(harassment):,})")

normal = df[df["is_nonbully"]][[text_col]].copy()
normal["label"] = 0
normal.to_csv(args.out_norm, index=False)
print(f"Non-bullying rows saved → {args.out_norm}  ({len(normal):,})")

print("✅  Done.")