"""
Extract ASA defamation rulings into a single CSV with columns:

title | background | ad_description | issue | response | assessment | action
"""

import re, time, requests, pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
LINKS_CSV   = Path("/Users/arlankalin/Downloads/Cyberbullying-emotion-main/data/asa_defam_cases.csv")      
HTML_CACHE  = Path("/Users/arlankalin/Downloads/Cyberbullying-emotion-main/data/asa_defam_pages")     
OUTPUT_CSV  = Path("/Users/arlankalin/Downloads/Cyberbullying-emotion-main/data/asa_defam_structured.csv")
DELAY       = 1.0                           
HEADERS     = {"User-Agent": "Mozilla/5.0 (research bot)"}
# ------------------------------------------------------------------

SECTION_MAP = {
    "background":        "background",
    "ad description":    "ad_description",
    "issue":             "issue",
    "issues":            "issue",
    "response":          "response",
    "responses":         "response",
    "assessment":        "assessment",
    "action":            "action",
}

def clean(txt: str) -> str:
    """Collapse whitespace & nbsp."""
    return re.sub(r"\s+", " ", txt.replace("\xa0", " ")).strip()

def fetch_html(url: str) -> str:
    """Download page (or reuse cached copy)."""
    HTML_CACHE.mkdir(exist_ok=True)
    fname = HTML_CACHE / (url.split("/")[-1] or "index.html")
    if fname.exists():
        return fname.read_text(encoding="utf-8")
    html = requests.get(url, headers=HEADERS, timeout=15).text
    fname.write_text(html, encoding="utf-8")
    time.sleep(DELAY)
    return html

rows = []
link_df = pd.read_csv(LINKS_CSV)          # ← expects column “link” or “Link”
col_name = "link" if "link" in link_df.columns else "Link"

for i, url in enumerate(link_df[col_name], 1):
    print(f"[{i}/{len(link_df)}]  {url}")
    soup = BeautifulSoup(fetch_html(url), "html.parser")

    entry = {c: "" for c in [
        "title", "background", "ad_description",
        "issue", "response", "assessment", "action"
    ]}
    entry["url"] = url
    entry["title"] = clean(soup.find("h1").get_text()) if soup.find("h1") else ""

    for h2 in soup.select("h2.font-color-grey"):
        heading = clean(h2.get_text()).lower()
        key = SECTION_MAP.get(heading)
        if not key:
            continue
        paras = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            if sib.name == "p":
                paras.append(clean(sib.get_text(" ", strip=True)))
        entry[key] = " ".join(paras)

    rows.append(entry)

df = pd.DataFrame(rows)

# keep the first occurrence of each URL (or title)
df = df.drop_duplicates(subset="url")      #  ← or subset="title"

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  ({len(df)} unique rulings)")
