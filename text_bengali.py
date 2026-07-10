#!/usr/bin/env python3
"""
Ishara — Bengali to Banglish (Latin) transliteration for the OLED (Phase 3c-2).

Extracted verbatim from robot_phase4.py. The only public name is bn_to_latin;
robot_phase4.py now does `from text_bengali import bn_to_latin` in place of the
old inline block. The mapping tables and algorithm are unchanged, so output is
identical to before.
"""

_BN_VOWEL = {'অ':'o','আ':'a','ই':'i','ঈ':'i','উ':'u','ঊ':'u','ঋ':'ri','এ':'e','ঐ':'oi','ও':'o','ঔ':'ou'}
_BN_CONS = {'ক':'k','খ':'kh','গ':'g','ঘ':'gh','ঙ':'ng','চ':'ch','ছ':'chh','জ':'j','ঝ':'jh','ঞ':'n',
            'ট':'t','ঠ':'th','ড':'d','ঢ':'dh','ণ':'n','ত':'t','থ':'th','দ':'d','ধ':'dh','ন':'n',
            'প':'p','ফ':'ph','ব':'b','ভ':'bh','ম':'m','য':'j','র':'r','ল':'l','শ':'sh','ষ':'sh',
            'স':'s','হ':'h','ড়':'r','ঢ়':'rh','য়':'y','ৎ':'t'}
_BN_MATRA = {'া':'a','ি':'i','ী':'i','ু':'u','ূ':'u','ৃ':'ri','ে':'e','ৈ':'oi','ো':'o','ৌ':'ou'}
_BN_OTHER = {'ং':'ng','ঃ':'h','ঁ':'n','়':''}
_BN_DIGIT = {'০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','৯':'9'}
_HASANTA = '্'


def bn_to_latin(s):
    out = []; ch = list(s); i = 0; n = len(ch)
    while i < n:
        c = ch[i]
        if c in _BN_CONS:
            out.append(_BN_CONS[c])
            nxt = ch[i + 1] if i + 1 < n else ''
            if nxt == _HASANTA:           # conjunct: drop inherent vowel, join next
                i += 2; continue
            if nxt in _BN_MATRA:
                out.append(_BN_MATRA[nxt]); i += 2; continue
            last = (i + 1 >= n) or (ch[i + 1] == ' ')
            if not last: out.append('o')  # inherent vowel
            i += 1; continue
        if c in _BN_VOWEL: out.append(_BN_VOWEL[c]); i += 1; continue
        if c in _BN_MATRA: out.append(_BN_MATRA[c]); i += 1; continue
        if c in _BN_OTHER: out.append(_BN_OTHER[c]); i += 1; continue
        if c in _BN_DIGIT: out.append(_BN_DIGIT[c]); i += 1; continue
        if c == ' ': out.append(' '); i += 1; continue
        if ord(c) < 128: out.append(c)
        i += 1
    return ''.join(out)
