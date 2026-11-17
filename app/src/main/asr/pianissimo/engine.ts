import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createBoost, type Boost } from './boost'
import { createTokenizer, type Tokenizer } from './bpe'
import { tdtDecode, type Joint } from './decode'

/**
 * Pianissimo (KlangAI/pianissimo-sv-onnx): Swedish speech to text, a Parakeet TDT run with onnxruntime on the processor.
 * Three networks in a row: the preprocessor (audio to 128 features per 10 ms), the encoder (one vector per 80 ms) and the
 * decoder-joint network asked once per step of the decoding loop (decode.ts). All files are in one folder (see FILES).
 * Measured on a 16-core laptop: 4 threads read audio about 25 times faster than it plays.
 */

/** the files of one build (int8: 660 MB) that the engine reads, next to each other in a folder */
export const FILES = (quant: string) => ({
  preprocessor: 'nemo128.onnx',
  encoder: `encoder-model.${quant}.onnx`,
  decoder: `decoder_joint-model.${quant}.onnx`,
  vocab: 'vocab.txt',
  tokenizer: 'tokenizer.model',
})

/** the part of onnxruntime-node used here, so this file does not need the package to be type-checked or tested */
export type Ort = {
  InferenceSession: { create(path: string, o?: object): Promise<Session> }
  Tensor: new (type: string, data: ArrayBufferView | BigInt64Array, dims: number[]) => Tensor
}
type Tensor = { data: Float32Array | BigInt64Array | Int32Array; dims: readonly number[] }
type Session = { run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>; release?(): Promise<void> }

/** how many threads onnxruntime may use: 4 was fastest on a 16-core laptop; more only fights over the cores */
export const THREADS = 4
/** seconds of audio one encoder frame stands for: 10 ms features, subsampled 8 times */
export const FRAME_S = 0.08
const MAX_PER_STEP = 10
const STATE = 2 * 640

export type Recognized = {
  text: string
  /** the words said, with times in ms from the start of the audio */
  words: Array<{ text: string; startMs: number; endMs: number }>
}

export type Engine = {
  /** the audio is 16 kHz mono, as the chunker makes it; phrases are the person's own words, to be boosted */
  recognize(samples: Float32Array, o?: { phrases?: string[]; alpha?: number }): Promise<Recognized>
  close(): Promise<void>
}

export async function loadEngine(o: { dir: string; quant?: string; ort: Ort; threads?: number }): Promise<Engine> {
  const f = FILES(o.quant ?? 'int8')
  const { ort } = o
  const sess = (name: string) => ort.InferenceSession.create(join(o.dir, name), { executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: o.threads ?? THREADS })
  const [pre, enc, dec] = await Promise.all([sess(f.preprocessor), sess(f.encoder), sess(f.decoder)])
  const vocab = readVocab(readFileSync(join(o.dir, f.vocab), 'utf8'))
  const tokenizer = createTokenizer(readFileSync(join(o.dir, f.tokenizer)))
  const trees = boostCache(tokenizer, vocab.blank)

  // one reading at a time: the encoder is the heavy part, and onnxruntime already uses the threads
  let queue: Promise<unknown> = Promise.resolve()
  const serial = <T>(fn: () => Promise<T>): Promise<T> => { const r = queue.then(fn); queue = r.catch(() => undefined); return r }

  async function read(samples: Float32Array, phrases: string[], alpha: number): Promise<Recognized> {
    if (samples.length < 1600) return { text: '', words: [] }
    const feat = await pre.run({ waveforms: new ort.Tensor('float32', samples, [1, samples.length]), waveforms_lens: new ort.Tensor('int64', BigInt64Array.from([BigInt(samples.length)]), [1]) })
    const e = await enc.run({ audio_signal: feat.features, length: feat.features_lens })
    const out = e.outputs, frames = Number((e.encoded_lengths.data as BigInt64Array)[0]), dim = out.dims[1], stride = out.dims[2]
    const frame = new Float32Array(dim)
    const joint: Joint = async (t, prev, state) => {
      for (let d = 0; d < dim; d++) frame[d] = (out.data as Float32Array)[d * stride + t]
      const [s1, s2] = state as [Float32Array, Float32Array]
      const r = await dec.run({
        encoder_outputs: new ort.Tensor('float32', frame, [1, dim, 1]),
        targets: new ort.Tensor('int32', Int32Array.from([prev]), [1, 1]),
        target_length: new ort.Tensor('int32', Int32Array.from([1]), [1]),
        input_states_1: new ort.Tensor('float32', s1, [2, 1, 640]),
        input_states_2: new ort.Tensor('float32', s2, [2, 1, 640]),
      })
      const all = r.outputs.data as Float32Array
      let step = 0
      for (let i = 1; i < all.length - vocab.size; i++) if (all[vocab.size + i] > all[vocab.size + step]) step = i
      return { logits: all.subarray(0, vocab.size), step, state: [r.output_states_1.data, r.output_states_2.data] }
    }
    const tree = phrases.length ? trees.get(phrases) : undefined
    const { tokens, times } = await tdtDecode({ frames, blank: vocab.blank, maxPerStep: MAX_PER_STEP, joint, start: [new Float32Array(STATE), new Float32Array(STATE)], ...(tree ? { boost: { tree, alpha } } : {}) })
    return words(tokens, times, vocab.pieces, frames)
  }

  return {
    recognize: (samples, p) => serial(() => read(samples, p?.phrases ?? [], p?.alpha ?? 1)),
    close: async () => { await Promise.all([pre, enc, dec].map((s) => s.release?.())) },
  }
}

/** vocab.txt: one "token id" per line, a space written as the lower block; the blank is the last */
export function readVocab(text: string): { pieces: string[]; size: number; blank: number } {
  const pieces: string[] = []
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    const at = line.lastIndexOf(' ')
    pieces[Number(line.slice(at + 1))] = line.slice(0, at).replaceAll('▁', ' ')
  }
  const blank = pieces.findIndex((p) => p === '<blk>')
  if (blank !== pieces.length - 1) throw new Error('expected the blank token to be the last token of the vocabulary')
  return { pieces, size: pieces.length, blank }
}

/** a boosting tree for a list of phrases, built once for each list (it changes when people join a call) */
function boostCache(tokenizer: Tokenizer, vocab: number) {
  let key = '', tree: Boost | undefined
  return { get(phrases: string[]): Boost { const k = phrases.join('\n'); if (k !== key || !tree) { tree = createBoost(phrases.map((p) => tokenizer.encode(p)), vocab); key = k } return tree } }
}

/** the tokens as words: a token that starts with a space starts a word; a word ends where the next starts */
export function words(tokens: number[], times: number[], pieces: string[], frames: number): Recognized {
  const out: Recognized['words'] = []
  let text = ''
  tokens.forEach((tok, i) => {
    const piece = pieces[tok], ms = Math.round(times[i] * FRAME_S * 1000)
    text += piece
    if (piece.startsWith(' ') || !out.length) out.push({ text: piece.trimStart(), startMs: ms, endMs: ms })
    else out[out.length - 1].text += piece
    const next = tokens.length > i + 1 ? times[i + 1] : frames
    out[out.length - 1].endMs = Math.max(ms, Math.round(next * FRAME_S * 1000))
  })
  // a word ends where the next one starts, so the gaps between words do not belong to nobody
  for (let i = 0; i + 1 < out.length; i++) out[i].endMs = Math.min(out[i].endMs, out[i + 1].startMs)
  return { text: text.trimStart(), words: out.filter((w) => w.text) }
}
