import os, openpyxl
from banglatts import BanglaTTS

BASE = "/home/hudai/Desktop/thesis"
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")
OUT = os.path.join(BASE, "audio")
os.makedirs(OUT, exist_ok=True)

ws = openpyxl.load_workbook(WORD_XLSX).active
id2name = {int(r[0]): str(r[1]).strip()
           for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}

print(f"generating {len(id2name)} audio files...")
tts = BanglaTTS(save_location=os.path.join(BASE, "tts_model"))

for cid, word in id2name.items():
    path = os.path.join(OUT, f"{cid}.wav")
    if os.path.exists(path):
        continue
    try:
        tts.convert(word, path)
        print(f"  saved {cid}.wav  ({word})")
    except Exception as e:
        print(f"  FAILED {cid} ({word}): {e}")

print("\nDONE  all 102 word files in audio/")
