import openpyxl
from gtts import gTTS

# read a real Bengali word from your label file
ws = openpyxl.load_workbook("/home/hudai/Desktop/thesis/Word Label.xlsx").active
id2name = {int(r[0]): str(r[1]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}

word = id2name[0]          # class 0 = ??????
print("word from xlsx:", repr(word))   # should show real Bengali characters, not ???

gTTS(word, lang="bn").save("test.mp3")
print("TTS OK -> test.mp3")
