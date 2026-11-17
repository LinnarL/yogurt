# Swedish speech: Pianissimo

This fork can read Swedish meetings with [Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv), a Swedish speech model by
KlangAI (a fine-tune of NVIDIA's Parakeet TDT 0.6B v3, CC BY 4.0), instead of whisper. It runs on the processor with
ONNX Runtime; nothing leaves the computer.

## For people using it

1. Settings → Language → Spoken language: pick **Swedish** (also in the picker under a note).
2. Settings → Language → **Swedish speech model**: Download (about 0.7 GB). Yogurt checks the files, then times the model once on this
   computer. If it can't read speech twice as fast as it plays, the files are deleted and whisper keeps reading Swedish.
3. Words you add under **Jargon**, the people in the meeting and their companies are given to the model as words to prefer
   ("phrase boosting"), so names come out spelled as you typed them. Up to 300 words are used; whisper only sees the 20 or so that fit in its 500-character prompt.

Until the model is in, Swedish is read by whisper (told "sv"). Other languages always use whisper.

## How it fits in the code

| Part | Where |
|---|---|
| Engine (preprocessor, encoder, token-and-duration decoding, word times) | `app/src/main/asr/pianissimo/engine.ts`, `decode.ts` |
| Phrase boosting (port of KlangAI's `phrase_boost.py`) and its BPE tokenizer | `boost.ts`, `bpe.ts` |
| Same shape as whisper's transcriber: a chunk in, lines out | `pianissimo/index.ts` (`createPianissimoTranscriber`) |
| Which engine reads each chunk | `app/src/main/asr/index.ts` (`open()`) |
| The model's files: download, hash check, speed check | `pianissimo/model.ts`, `speed.ts`; the files are pinned in `catalog/catalog.json` (`speechSv`) |
| Engine kept loaded between meetings, let go after 10 minutes | `pianissimo/holder.ts` |
| Settings row and the `swedish:*` IPC | `app/src/renderer/src/components/SwedishModel.tsx`, `app/src/main/index.ts` |

## Trying it without the app

```bash
node app/scripts/pianissimo-download.mjs /some/folder          # downloads and verifies the files in the catalog
PIANISSIMO_MODEL=/some/folder pnpm -C app exec vitest run test/pianissimo
```

`PIANISSIMO_CLIPS` (a folder of 16 kHz wav files with a `refs.json` of `{ "file.wav": "what is said" }`) adds the tests that compare
text, and the live two-track meeting test (`PIANISSIMO_PACED=1` feeds it at real-time speed). Without them those tests skip.
The tokenizer and the boosting tree are checked against KlangAI's python through `app/test/fixtures/pianissimo-fixture.json`
(made by `app/scripts/pianissimo-fixture.py`).

## Measured (16-core laptop, int8, CPU, 4 threads)

- One track: about 25 times faster than real time. Two tracks with quick partial readings at a meeting's pace: finished at the moment the last audio arrived.
- DirectML (GPU) does not run these files: the encoder's block-local attention uses 5-dimensional matrix multiplications that DirectML rejects. CPU only.

## Following upstream

The fork changes `electron-builder.yml`'s `publish.owner` so installed copies update from this fork, not from FrigadeHQ's releases.
`git fetch upstream && git rebase upstream/main` brings in new releases; the changes outside `app/src/main/asr/pianissimo/` are small.
