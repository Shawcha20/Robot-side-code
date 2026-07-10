from banglatts import BanglaTTS
import openpyxl, os

# read a real Bengali word from your xlsx
ws = openpyxl.load_workbook("/home/hudai/Desktop/thesis/Word Label.xlsx").active
id2name = {int(r[0]): str(r[1]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}
word = id2name[2]   # ??????

bn = BanglaTTS(save_location='/home/hudai/Desktop/thesis/silero_model')
import time
t0 = time.time()
out_file = bn(word, filename="test_bangla.wav", voice='male')
print(f"generated in {time.time()-t0:.1f}s -> {out_file}")
