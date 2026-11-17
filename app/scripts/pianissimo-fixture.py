"""Writes app/test/fixtures/pianissimo-fixture.json: what KlangAI's python gives for some words, to check the TypeScript port against.

    python -I app/scripts/pianissimo-fixture.py <folder with tokenizer.model and phrase_boost.py> <output.json>

Needs: pip install sentencepiece numpy. The folder is a download of huggingface.co/KlangAI/pianissimo-sv-onnx (CC BY 4.0).
"""
import json
import sys

import sentencepiece as spm

folder, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, folder)
from phrase_boost import PhraseBoost  # noqa: E402

tok = f"{folder}/tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=tok)
phrases = ["Baba Shyam", "chhappan bhog", "hinduismen", "Ekologigruppen", "Länsstyrelsen", "NVI", "Naturvärdesinventering", "hasselmus",
           "Åke Öberg", "e-post 2026", "Skövde", "Ångström", "naturvårdsverket", "SS 199000:2023", "  extra   spaces  ", "ÖÖÖ😀", "x", "Frigade"]
boost_phrases = ["Baba Shyam", "chhappan bhog", "hinduismen", "Naturvärdesinventering", "Skövde"]
tree = PhraseBoost(boost_phrases, tok, vocab_size=8192)

# the states reachable from the start, breadth first (the first 14)
states, seen, todo = [], {0}, [0]
while todo and len(states) < 14:
    s = todo.pop(0)
    states.append(s)
    for t in set(tree.scores(s)[1].tolist()):
        if t != 0 and t not in seen:
            seen.add(t)
            todo.append(t)


def compact(a, digits):
    """the commonest value, and the entries that differ from it"""
    vals = [round(float(x), digits) for x in a]
    base = max(set(vals), key=vals.count)
    return {"base": base, "other": {i: v for i, v in enumerate(vals) if v != base}}


with open(out, "w", encoding="utf-8") as f:
    json.dump({
        "tokens": {p: sp.encode(p) for p in phrases},
        "boostPhrases": boost_phrases,
        "boostPhraseTokens": [sp.encode(p) for p in boost_phrases],
        "states": [{"state": s, "bonus": compact(tree.scores(s)[0], 5), "next": compact(tree.scores(s)[1], 0)} for s in states],
    }, f, ensure_ascii=False)
