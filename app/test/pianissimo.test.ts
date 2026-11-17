import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { createTokenizer, readModel } from '../src/main/asr/pianissimo/bpe'
import { createBoost } from '../src/main/asr/pianissimo/boost'
import { tdtDecode, type Joint } from '../src/main/asr/pianissimo/decode'
import { readVocab, words, type Engine } from '../src/main/asr/pianissimo/engine'
import { createPianissimoTranscriber } from '../src/main/asr/pianissimo/index'
import type { Chunk } from '../src/main/asr/chunker'

/**
 * Most of this runs on tiny made-up models. The checks against KlangAI's python (fixtures/pianissimo-fixture.json, made by
 * scripts/pianissimo-fixture.py) and against the real model run when PIANISSIMO_MODEL is the folder of a download of
 * huggingface.co/KlangAI/pianissimo-sv-onnx, and PIANISSIMO_CLIPS a folder of 16 kHz wav files with a refs.json.
 */
const MODEL = process.env.PIANISSIMO_MODEL
const fixture = JSON.parse(readFileSync(join(__dirname, 'fixtures/pianissimo-fixture.json'), 'utf8')) as {
  tokens: Record<string, number[]>; boostPhrases: string[]; boostPhraseTokens: number[][]
  states: Array<{ state: number; bonus: { base: number; other: Record<string, number> }; next: { base: number; other: Record<string, number> } }>
}

// --- a SentencePiece model file, built by hand: pieces, then the normalizer spec
const varint = (n: number) => { const b: number[] = []; while (n > 127) { b.push((n & 127) | 128); n = Math.floor(n / 128) } b.push(n); return b }
const bytesField = (no: number, data: number[]) => [...varint(no * 8 + 2), ...varint(data.length), ...data]
const piece = (text: string, score: number, type = 1) => {
  const f = new Float32Array([score])
  return bytesField(1, [...bytesField(1, [...new TextEncoder().encode(text)]), ...varint(2 * 8 + 5), ...new Uint8Array(f.buffer), ...varint(3 * 8), ...varint(type)])
}
const model = (pieces: Array<[string, number, number?]>, normalizer: number[] = []) => new Uint8Array([...pieces.flatMap(([t, s, ty]) => piece(t, s, ty)), ...bytesField(3, normalizer)])

describe('bpe', () => {
  // ids: 0 unk, 1 ▁, 2 a, 3 b, 4 ab, 5 ▁a, 6 ▁ab
  const pieces: Array<[string, number, number?]> = [['<unk>', 0, 2], ['▁', -1], ['a', -2], ['b', -3], ['ab', -2.5], ['▁a', -2.2], ['▁ab', -1.5]]
  it('reads the pieces in id order with their scores and types', () => {
    const m = readModel(model(pieces))
    expect(m.pieces.map((p) => p.text)).toEqual(['<unk>', '▁', 'a', 'b', 'ab', '▁a', '▁ab'])
    expect(m.pieces[4].score).toBeCloseTo(-2.5)
    expect(m.pieces[0].type).toBe(2)
    expect(m.addDummyPrefix).toBe(true)
  })
  it('merges the best scoring pair first, with a dummy space in front', () => {
    const t = createTokenizer(model(pieces))
    expect(t.encode('ab')).toEqual([6])
    expect(t.encode('ab b')).toEqual([6, 1, 3])
  })
  it('gives the unknown piece for a character it has no piece for', () => {
    expect(createTokenizer(model(pieces)).encode('az')).toEqual([5, 0])
  })
  it('keeps extra whitespace when the model says so, and removes it otherwise', () => {
    const keep = model(pieces, [...varint(4 * 8), 0])
    expect(createTokenizer(keep).encode('a  b')).toEqual([5, 1, 1, 3])
    expect(createTokenizer(model(pieces)).encode('a  b')).toEqual([5, 1, 3])
  })
  it('has nothing to say about nothing', () => expect(createTokenizer(model(pieces)).encode('   ')).toEqual([]))

  it.skipIf(!MODEL)('gives the tokens that the python sentencepiece gives for the real model', () => {
    const t = createTokenizer(readFileSync(join(MODEL!, 'tokenizer.model')))
    expect(t.size).toBe(8192)
    for (const [phrase, ids] of Object.entries(fixture.tokens)) expect(t.encode(phrase), phrase).toEqual(ids)
  })
})

describe('boost', () => {
  const at = (v: { base: number; other: Record<string, number> }, n: number) => { const a = new Float32Array(n).fill(v.base); for (const [i, x] of Object.entries(v.other)) a[+i] = x; return a }

  it('gives the same bonus and next states as the python port, for the states reachable from the start', () => {
    const boost = createBoost(fixture.boostPhraseTokens, 8192)
    for (const s of fixture.states) {
      const got = boost.scores(s.state)
      const bonus = at(s.bonus, 8192), next = at(s.next, 8192)
      for (let i = 0; i < 8192; i++) {
        if (Math.abs(got.bonus[i] - bonus[i]) > 1e-4) throw new Error(`state ${s.state} token ${i}: bonus ${got.bonus[i]}, python ${bonus[i]}`)
        if (got.next[i] !== next[i]) throw new Error(`state ${s.state} token ${i}: next ${got.next[i]}, python ${next[i]}`)
      }
    }
  })
  it('helps the next token of a phrase and falls back to the start for others', () => {
    const b = createBoost([[5, 6, 7]], 10)
    const start = b.scores(0)
    expect(start.bonus[5]).toBeCloseTo(1)
    expect(start.bonus[3]).toBe(0)
    expect(start.next[3]).toBe(0)
    const inside = b.scores(start.next[5])
    expect(inside.bonus[6]).toBeGreaterThan(2)
    // a wrong turn in the middle gives back what was earned so far
    expect(inside.bonus[3]).toBeLessThan(0)
  })
  it('ignores empty and repeated phrases', () => {
    const b = createBoost([[], [5, 6], [5, 6]], 10)
    expect(b.scores(0).bonus[5]).toBeCloseTo(1)
  })
})

describe('tdtDecode', () => {
  // a joint that follows a script: for each frame t, the token to say and the duration to move by
  const scripted = (script: Record<number, Array<{ token: number; step: number }>>, blank = 9, size = 10): { joint: Joint; asked: Array<[number, number]> } => {
    const asked: Array<[number, number]> = []
    const calls: Record<number, number> = {}
    const joint: Joint = async (t, prev, state) => {
      asked.push([t, prev])
      const s = script[t]?.[calls[t] = (calls[t] ?? -1) + 1] ?? { token: blank, step: 1 }
      const logits = new Float32Array(size); logits[s.token] = 5
      return { logits, step: s.step, state: (state as number) + 1 }
    }
    return { joint, asked }
  }

  it('writes down tokens that are not the blank, and skips frames by the duration', async () => {
    const { joint, asked } = scripted({ 0: [{ token: 3, step: 2 }], 2: [{ token: 4, step: 1 }] })
    const r = await tdtDecode({ frames: 5, blank: 9, maxPerStep: 10, joint, start: 0 })
    expect(r).toEqual({ tokens: [3, 4], times: [0, 2] })
    // the token before is the blank at the start, then the last token said
    expect(asked.map(([t, p]) => `${t}:${p}`)).toEqual(['0:9', '2:3', '3:4', '4:4'])
  })
  it('can say several tokens on one frame (duration 0), up to maxPerStep, then moves on', async () => {
    const { joint } = scripted({ 0: [{ token: 1, step: 0 }, { token: 2, step: 0 }, { token: 3, step: 0 }] })
    const r = await tdtDecode({ frames: 2, blank: 9, maxPerStep: 2, joint, start: 0 })
    expect(r.tokens).toEqual([1, 2])
    expect(r.times).toEqual([0, 0])
  })
  it('lets the bonus choose another token than the model would, but never the blank', async () => {
    const joint: Joint = async () => { const l = new Float32Array(10); l[3] = 2; l[4] = 1.5; return { logits: l, step: 1, state: 0 } }
    const plain = await tdtDecode({ frames: 1, blank: 9, maxPerStep: 10, joint, start: 0 })
    expect(plain.tokens).toEqual([3])
    const boosted = await tdtDecode({ frames: 1, blank: 9, maxPerStep: 10, joint, start: 0, boost: { tree: createBoost([[4]], 9), alpha: 1 } })
    expect(boosted.tokens).toEqual([4])
    const none = await tdtDecode({ frames: 1, blank: 9, maxPerStep: 10, joint: async () => { const l = new Float32Array(10); l[9] = 3; l[4] = 1; return { logits: l, step: 1, state: 0 } }, start: 0, boost: { tree: createBoost([[4]], 9), alpha: 100 } })
    expect(none.tokens).toEqual([])
  })
  it('follows the phrase through the boosting tree: the second token is helped after the first', async () => {
    const seen: number[] = []
    const joint: Joint = async (t) => { const l = new Float32Array(10); l[1] = 1; l[t === 0 ? 5 : 6] = 0.9; seen.push(t); return { logits: l, step: 1, state: 0 } }
    const r = await tdtDecode({ frames: 2, blank: 9, maxPerStep: 10, joint, start: 0, boost: { tree: createBoost([[5, 6]], 9), alpha: 1 } })
    expect(r.tokens).toEqual([5, 6])
  })
})

describe('words', () => {
  const pieces = ['▁', ' hej', ' då', '.', 'x'].map((p) => p.replace('▁', ' '))
  it('starts a word at a token with a space in front and ends it where the next starts', () => {
    const r = words([1, 2, 3], [0, 3, 5], pieces, 8)
    expect(r.text).toBe('hej då.')
    expect(r.words).toEqual([{ text: 'hej', startMs: 0, endMs: 240 }, { text: 'då.', startMs: 240, endMs: 640 }])
  })
  it('has nothing for no tokens', () => expect(words([], [], pieces, 4)).toEqual({ text: '', words: [] }))
})

describe('vocabulary', () => {
  it('reads "token id" lines, the space as the lower block, and wants the blank last', () => {
    const v = readVocab('<unk> 0\n▁hej 1\n<blk> 2\n')
    expect(v.pieces).toEqual(['<unk>', ' hej', '<blk>'])
    expect(v.blank).toBe(2)
    expect(() => readVocab('<blk> 0\nx 1\n')).toThrow(/blank/)
  })
})

describe('transcriber', () => {
  const chunk: Chunk = { track: 'me', startMs: 10_000, samples: new Float32Array(16000) }
  const engineSaying = (r: Awaited<ReturnType<Engine['recognize']>>, seen: Array<{ phrases?: string[]; alpha?: number }> = []): Engine => ({ recognize: async (_s, o) => { seen.push(o ?? {}); return r }, close: async () => undefined })
  const said = { text: 'Hej då. Vi ses', words: [{ text: 'Hej', startMs: 100, endMs: 400 }, { text: 'då.', startMs: 400, endMs: 800 }, { text: 'Vi', startMs: 1000, endMs: 1200 }, { text: 'ses', startMs: 1200, endMs: 1600 }] }

  it('gives a line for each sentence on meeting time, with the words', async () => {
    const heard: unknown[] = []
    const t = createPianissimoTranscriber({ engine: async () => engineSaying(said), onLanguage: (...a) => heard.push(a) })
    const lines = await t(chunk)
    expect(lines.map((l) => [l.text, l.startMs, l.endMs])).toEqual([['Hej då.', 10_100, 10_800], ['Vi ses', 11_000, 11_600]])
    expect(lines[0].words?.[1]).toEqual({ text: 'då.', startMs: 10_400, endMs: 10_800 })
    expect(heard).toEqual([['me', 'swedish', 14, true]])
  })
  it('hands the phrases to the engine, read for each chunk', async () => {
    const seen: Array<{ phrases?: string[]; alpha?: number }> = []
    let list = ['Kalle']
    const t = createPianissimoTranscriber({ engine: async () => engineSaying(said, seen), phrases: () => list, alpha: () => 2 })
    await t(chunk); list = ['Kalle', 'Anna']; await t(chunk)
    expect(seen).toEqual([{ phrases: ['Kalle'], alpha: 2 }, { phrases: ['Kalle', 'Anna'], alpha: 2 }])
  })
  it('says nothing for silence or garble, and does not count a partial towards the language', async () => {
    const heard: unknown[] = []
    expect(await createPianissimoTranscriber({ engine: async () => engineSaying({ text: '', words: [] }) })(chunk)).toEqual([])
    const garble = 'ja ja ja ja ja ja ja ja ja ja'
    expect(await createPianissimoTranscriber({ engine: async () => engineSaying({ text: garble, words: [{ text: garble, startMs: 0, endMs: 1 }] }) })(chunk)).toEqual([])
    const partial = createPianissimoTranscriber({ engine: async () => engineSaying(said), onLanguage: (...a) => heard.push(a) })
    expect((await partial(chunk, { partial: true })).length).toBe(2)
    expect(heard).toEqual([])
  })
})
