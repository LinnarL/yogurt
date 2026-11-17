import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { loadEngine, type Engine, type Ort } from '../src/main/asr/pianissimo/engine'

/**
 * The real model, on the processor. Runs only where PIANISSIMO_MODEL is the folder of a download of
 * huggingface.co/KlangAI/pianissimo-sv-onnx (int8 files), PIANISSIMO_CLIPS a folder with 16 kHz wav files and a refs.json
 * ({ "file.wav": "what is said" }), and onnxruntime-node can be found from here. A few seconds of audio takes a few seconds.
 */
const MODEL = process.env.PIANISSIMO_MODEL, CLIPS = process.env.PIANISSIMO_CLIPS

/** 16 kHz mono wav, 16 bit or float */
function readWav(buf: Buffer): Float32Array {
  let off = 12, bits = 16, fmt = 1
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4), size = buf.readUInt32LE(off + 4)
    if (id === 'fmt ') { fmt = buf.readUInt16LE(off + 8); bits = buf.readUInt16LE(off + 22) }
    if (id === 'data') {
      const bps = bits / 8, n = Math.floor(Math.min(size, buf.length - off - 8) / bps), out = new Float32Array(n)
      for (let i = 0; i < n; i++) { const p = off + 8 + i * bps; out[i] = fmt === 3 ? buf.readFloatLE(p) : bits === 16 ? buf.readInt16LE(p) / 32768 : buf.readInt32LE(p) / 2147483648 }
      return out
    }
    off += 8 + size + (size % 2)
  }
  throw new Error('no data chunk')
}
/** what is said, without capitals, commas and full stops, to compare with the plain text of a reference */
const plain = (s: string) => s.toLowerCase().replace(/[.,?!]/g, '').replace(/\s+/g, ' ').trim()

/** the bundled speed-check clip (English): the model need not understand it, only run through it, fast */
describe.skipIf(!MODEL)('Pianissimo, the real model, on the bundled clip', () => {
  it('runs through a clip faster than it plays, with words in time order', async () => {
    const modName = 'onnxruntime-node'
    const engine = await loadEngine({ dir: MODEL!, ort: (await import(/* @vite-ignore */ modName)) as Ort })
    try {
      const clip = readWav(readFileSync(join(__dirname, '../resources/speed/clip.wav')))
      await engine.recognize(clip.subarray(0, 3 * 16000))
      const t0 = performance.now()
      const r = await engine.recognize(clip)
      const rtf = (performance.now() - t0) / 1000 / (clip.length / 16000)
      expect(r.words.length).toBeGreaterThan(5)
      r.words.forEach((w, i) => { if (i) expect(w.startMs).toBeGreaterThanOrEqual(r.words[i - 1].startMs) })
      // the app keeps the model only where this is at most 0.5; a CI runner has fewer cores than a laptop, so the bar here is the same
      expect(rtf).toBeLessThan(0.5)
    } finally { await engine.close() }
  }, 120_000)
})

describe.skipIf(!MODEL || !CLIPS)('Pianissimo, the real model', () => {
  let engine: Engine
  const refs = CLIPS ? (JSON.parse(readFileSync(join(CLIPS, 'refs.json'), 'utf8')) as Record<string, string>) : {}
  const names = Object.keys(refs)
  const audio = (n: string) => readWav(readFileSync(join(CLIPS!, n)))
  beforeAll(async () => {
    const modName = 'onnxruntime-node'
    engine = await loadEngine({ dir: MODEL!, ort: (await import(/* @vite-ignore */ modName)) as Ort })
  }, 120_000)
  afterAll(() => engine?.close())

  it('reads Swedish speech, with capitals and full stops', async () => {
    const r = await engine.recognize(audio(names[0]))
    expect(plain(r.text)).toBe(plain(refs[names[0]]))
    expect(r.text[0]).toBe(r.text[0].toUpperCase())
    expect(r.text.endsWith('.')).toBe(true)
    // the words have times that go forward and stay inside the clip
    const clipMs = (audio(names[0]).length / 16000) * 1000
    expect(r.words.length).toBeGreaterThan(10)
    r.words.forEach((w, i) => { expect(w.endMs).toBeGreaterThanOrEqual(w.startMs); expect(w.endMs).toBeLessThanOrEqual(clipMs + 100); if (i) expect(w.startMs).toBeGreaterThanOrEqual(r.words[i - 1].startMs) })
  }, 60_000)

  it('reads the other clips the same as the reference, but for names it cannot know', async () => {
    for (const n of names.slice(1, 4)) expect(plain((await engine.recognize(audio(n))).text), n).toBe(plain(refs[n]))
  }, 120_000)

  it('writes a name as the person spelled it when it is in the phrase list', async () => {
    const n = names.find((x) => /baba shyam/.test(refs[x]))
    if (!n) return
    const without = (await engine.recognize(audio(n))).text
    const boosted = (await engine.recognize(audio(n), { phrases: ['Baba Shyam'] })).text
    expect(without).not.toContain('Baba Shyam')
    expect(boosted).toContain('Baba Shyam')
  }, 120_000)

  it('gives nothing for silence and for less than a tenth of a second', async () => {
    expect((await engine.recognize(new Float32Array(16000 * 2))).text).toBe('')
    expect((await engine.recognize(new Float32Array(800))).words).toEqual([])
  }, 60_000)

  it('reads two things asked at once one after the other', async () => {
    const [a, b] = await Promise.all([engine.recognize(audio(names[0])), engine.recognize(audio(names[1]))])
    expect(plain(a.text)).toBe(plain(refs[names[0]]))
    expect(plain(b.text)).toBe(plain(refs[names[1]]))
  }, 120_000)
})
