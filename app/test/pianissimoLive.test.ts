import { EventEmitter } from 'node:events'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { afterEach, describe, expect, it } from 'vitest'
import { Capture, encodeFrame, hookCapture, recordingFinished, type CaptureProc } from '../src/main/capture'
import { hookTranscription, settleTranscription, stopTranscriptionServer } from '../src/main/asr'
import { createEngineHolder } from '../src/main/asr/pianissimo/holder'
import type { Ort } from '../src/main/asr/pianissimo/engine'
import { openStore, type Meeting } from '../src/main/store'
import type { TranscriptPartial } from '../src/shared/types'

/**
 * A whole meeting through the live pipeline with the real model: two tracks of Swedish speech at once, quick partial readings
 * on, all of it pushed as fast as the pipeline takes it. It has to keep up (finish in under half the audio's length: the
 * bar the speed check uses) and read what was said. PIANISSIMO_PACED=1 feeds it at the pace of a real meeting (as long as the
 * audio), which is when the partial readings happen. Runs only where PIANISSIMO_MODEL (the model's folder) and
 * PIANISSIMO_CLIPS (16 kHz wav files and a refs.json) are set.
 */
const MODEL = process.env.PIANISSIMO_MODEL, CLIPS = process.env.PIANISSIMO_CLIPS
const RATE = 16000

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
const words = (s: string) => s.toLowerCase().replace(/[.,?!]/g, '').split(/\s+/).filter(Boolean)

const disposers: Array<() => void> = []
afterEach(async () => { disposers.splice(0).forEach((d) => d()); await settleTranscription(5000); stopTranscriptionServer() })

describe.skipIf(!MODEL || !CLIPS)('a Swedish meeting through the live pipeline, with the real model', () => {
  it('keeps up with two tracks at once, partial readings on, and reads what was said', async () => {
    const refs = JSON.parse(readFileSync(join(CLIPS!, 'refs.json'), 'utf8')) as Record<string, string>
    const names = Object.keys(refs)
    const clips = names.map((n) => readWav(readFileSync(join(CLIPS!, n))))
    const gap = new Float32Array(Math.round(1.6 * RATE))
    // me: the clips 0, 2, 4; them: the clips 1, 3, 5 (a pause after each so the chunker cuts there)
    const track = (ix: number[]) => Float32Array.from(ix.flatMap((i) => [...gap, ...clips[i]]).concat([...gap]))
    const audio = { me: track([0, 2, 4]), them: track([1, 3, 5]) }
    const lengthS = Math.max(audio.me.length, audio.them.length) / RATE

    const store = openStore(':memory:')
    const lines: Array<{ track: string; text: string }> = []
    const partials: TranscriptPartial[] = []
    const starts: Array<(m: Meeting) => void> = [], stops: Array<(m: Meeting) => void> = []
    const onStart = (fn: (m: Meeting) => void) => { starts.push(fn); return () => {} }
    const onStop = (fn: (m: Meeting) => void) => { stops.push(fn); return () => {} }
    const out = new PassThrough(), stdin = new PassThrough(), ev = new EventEmitter()
    stdin.on('data', (d: Buffer) => { if (d.toString().includes('stop')) setImmediate(() => ev.emit('exit', 0, null)) })
    const proc: CaptureProc = { stdout: out, stdin, on: (e, fn) => { ev.on(e, fn); return proc }, kill: () => true }
    const modName = 'onnxruntime-node'
    const holder = createEngineHolder({ ort: async () => (await import(/* @vite-ignore */ modName)) as Ort, dir: () => MODEL })
    disposers.push(hookTranscription({
      store, modelPath: () => '/none.bin', send: (s) => lines.push({ track: s.track, text: s.text }), sendPartial: (p) => partials.push(p), warn: (c) => { throw new Error(`warned: ${c}`) },
      onStart, onStop, pidFile: undefined, spoken: () => 'sv', swedish: { ready: () => true, engine: () => holder.get() },
    }), holder.stop)
    hookCapture({ dirOf: (id) => `/m/${id}`, onStart, onStop, stopMeeting: () => {}, join: async () => {}, start: () => new Capture(proc) })
    const m = store.createMeeting({ title: 'Möte', startedAt: 1, endedAt: null, eventId: null, app: null, attendees: [], state: 'recording' })
    starts.forEach((f) => f(m))

    // warm the engine first (loading it takes a few seconds and is not what is timed)
    await holder.get()
    const t0 = performance.now()
    for (let i = 0; i < audio.me.length; i += 1600) {
      for (const t of ['me', 'them'] as const) if (i < audio[t].length) out.write(encodeFrame({ t: 'pcm', track: t, ms: i / 16, samples: audio[t].subarray(i, i + 1600) }))
      // paced: the recorder's own speed (1600 samples are 100 ms); otherwise as fast as the pipeline takes it
      await new Promise((r) => setTimeout(r, process.env.PIANISSIMO_PACED ? 100 : i % 160000 === 0 ? 0 : -1))
    }
    const fed = (performance.now() - t0) / 1000
    stops.forEach((f) => f(m)); ev.emit('exit', 0, null)
    await recordingFinished(m.id); await settleTranscription(120_000)
    const took = (performance.now() - t0) / 1000

    const said = (t: string) => lines.filter((l) => l.track === t).map((l) => l.text).join(' ')
    const want = (ix: number[]) => ix.map((i) => refs[names[i]]).join(' ')
    const hit = (got: string, ref: string) => { const g = new Set(words(got)); const r = words(ref); return r.filter((w) => g.has(w)).length / r.length }
    console.log(`live Swedish meeting: ${lengthS.toFixed(0)} s of audio on 2 tracks, fed in ${fed.toFixed(1)} s, all read ${(took - fed).toFixed(1)} s after the last of it (${took.toFixed(1)} s in all), ${lines.length} lines, ${partials.length} partial readings`)
    expect(hit(said('me'), want([0, 2, 4]))).toBeGreaterThan(0.9)
    expect(hit(said('them'), want([1, 3, 5]))).toBeGreaterThan(0.9)
    // paced, the time is the audio's own length; what counts then is how long it took after the last word
    if (!process.env.PIANISSIMO_PACED) expect(took / lengthS).toBeLessThan(0.5)
    else expect(took - fed).toBeLessThan(3)
  }, 300_000)
})
