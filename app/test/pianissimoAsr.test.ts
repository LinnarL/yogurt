import { EventEmitter } from 'node:events'
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { PassThrough } from 'node:stream'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Capture, encodeFrame, hookCapture, recordingFinished, type CaptureProc } from '../src/main/capture'
import { hookTranscription, settleTranscription, stopTranscriptionServer } from '../src/main/asr'
import type { Engine } from '../src/main/asr/pianissimo/engine'
import { setJargon, setOwnerName, boostTerms, meetingWords } from '../src/main/vocab'
import type { Spoken } from '../src/shared/spoken'
import { openStore, type Meeting, type Store } from '../src/main/store'

/**
 * Which engine reads a Swedish meeting: Pianissimo once it is installed and fast enough, whisper (told Swedish) before that, and
 * whisper for every other language. The engines here are stand-ins; what is checked is who gets which chunk.
 */
const RATE = 16000
const tone = (sec: number) => Float32Array.from({ length: Math.round(sec * RATE) }, (_, i) => 0.3 * Math.sin(i / 10))

function fakeProc() {
  const ev = new EventEmitter(), out = new PassThrough(), stdin = new PassThrough()
  stdin.on('data', (d: Buffer) => { if (d.toString().includes('stop')) setImmediate(() => ev.emit('exit', 0, null)) })
  const proc: CaptureProc = { stdout: out, stdin, on: (e, fn) => { ev.on(e, fn); return proc }, kill: () => { ev.emit('exit', null, 'SIGKILL'); return true } }
  return {
    proc, exit: (code: number | null) => ev.emit('exit', code, null),
    say: (track: 'me' | 'them', ms: number) => {
      const all = Float32Array.from([...tone(1), ...new Float32Array(RATE)])
      for (let i = 0; i < all.length; i += 1600) out.write(encodeFrame({ t: 'pcm', track, ms: ms + i / 16, samples: all.subarray(i, i + 1600) }))
    },
  }
}

/** a stand-in whisper-server that answers every request with one line */
let http: Server | undefined
async function whisperStub(requests: Array<Record<string, string>>) {
  http = createServer((req, res) => {
    const parts: Buffer[] = []
    req.on('data', (d: Buffer) => parts.push(d))
    req.on('end', () => {
      const fields: Record<string, string> = {}
      for (const m of Buffer.concat(parts).toString('latin1').matchAll(/name="([^"]+)"(?!; filename)\r\n\r\n([^\r]*)\r\n/g)) fields[m[1]] = m[2]
      requests.push(fields)
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ detected_language: 'swedish', detected_language_probability: 0.99, segments: [{ start: 0.1, end: 1, text: ' från whisper', avg_logprob: -0.1, no_speech_prob: 0.01 }] }))
    })
  })
  await new Promise<void>((r) => http!.listen(0, '127.0.0.1', r))
  return `http://127.0.0.1:${(http.address() as AddressInfo).port}`
}

let store: Store
beforeEach(() => { store = openStore(':memory:'); setJargon(() => []); setOwnerName(() => '') })
const disposers: Array<() => void> = []
afterEach(async () => {
  disposers.splice(0).forEach((d) => d())
  await settleTranscription(2000); stopTranscriptionServer()
  await new Promise<void>((r) => (http ? http.close(() => r()) : r())); http = undefined
})

async function rig(o: { spoken: () => Spoken; ready: () => boolean; engine?: () => Promise<Engine>; attendees?: Meeting['attendees'] }) {
  const requests: Array<Record<string, string>> = []
  const url = await whisperStub(requests)
  const phrases: Array<string[] | undefined> = []
  const engine: Engine = { close: async () => undefined, recognize: async (_s, p) => { phrases.push(p?.phrases); return { text: 'Hej då.', words: [{ text: 'Hej', startMs: 0, endMs: 300 }, { text: 'då.', startMs: 300, endMs: 700 }] } } }
  const made: string[] = []
  const lines: string[] = []
  const starts: Array<(m: Meeting) => void> = [], stops: Array<(m: Meeting) => void> = []
  const onStart = (fn: (m: Meeting) => void) => { starts.push(fn); return () => {} }
  const onStop = (fn: (m: Meeting) => void) => { stops.push(fn); return () => {} }
  const procs: ReturnType<typeof fakeProc>[] = []
  disposers.push(hookTranscription({
    store, modelPath: () => '/model.bin', send: (s) => lines.push(`${s.track}:${s.text}`), warn: () => {}, onStart, onStop, pidFile: undefined, holdMs: 10, extraWaitMs: 10,
    spoken: o.spoken, swedish: { ready: o.ready, engine: o.engine ?? (async () => engine) },
    server: () => { made.push('whisper-server'); return { url: async () => url, resetBudget: () => {}, wedged: async () => url, alive: async () => true, kill: () => {}, stop: async () => {} } as never },
  }))
  hookCapture({ dirOf: (id) => `/m/${id}`, onStart, onStop, stopMeeting: () => {}, join: async () => {}, start: () => { const f = fakeProc(); procs.push(f); return new Capture(f.proc) } })
  const m = store.createMeeting({ title: 'Möte', startedAt: 1, endedAt: null, eventId: null, app: null, attendees: o.attendees ?? [], state: 'recording' })
  starts.forEach((f) => f(m))
  return {
    m, lines, requests, phrases, made, say: (ms = 0) => procs[0].say('me', ms),
    stop: async () => { stops.forEach((f) => f(m)); procs[0].exit(0); await recordingFinished(m.id); await settleTranscription(3000) },
  }
}

describe('who reads a meeting', () => {
  it('Pianissimo reads Swedish when it is installed: whisper-server is never started', async () => {
    const r = await rig({ spoken: () => 'sv', ready: () => true })
    r.say()
    await expect.poll(() => r.lines.length, { timeout: 5000 }).toBeGreaterThan(0)
    await r.stop()
    expect(r.lines[0]).toBe('me:Hej då.')
    expect(r.made).toEqual([])
    expect(r.requests).toEqual([])
  })

  it('whisper reads Swedish, told so, until Pianissimo is in', async () => {
    const r = await rig({ spoken: () => 'sv', ready: () => false })
    r.say()
    await expect.poll(() => r.lines.length, { timeout: 5000 }).toBeGreaterThan(0)
    await r.stop()
    expect(r.lines[0]).toBe('me:från whisper')
    expect(r.requests.every((q) => q.language === 'sv')).toBe(true)
    expect(r.phrases).toEqual([])
  })

  it('whisper reads the other languages, even where Pianissimo is installed', async () => {
    const r = await rig({ spoken: () => 'multi', ready: () => true })
    r.say()
    await expect.poll(() => r.lines.length, { timeout: 5000 }).toBeGreaterThan(0)
    await r.stop()
    expect(r.lines[0]).toBe('me:från whisper')
    expect(r.phrases).toEqual([])
  })

  it('follows the picker under a meeting, chunk by chunk, and starts whisper only when it is needed', async () => {
    let spoken: Spoken = 'sv'
    const r = await rig({ spoken: () => spoken, ready: () => true })
    r.say(0)
    await expect.poll(() => r.lines.length, { timeout: 5000 }).toBe(1)
    expect(r.made).toEqual([])
    spoken = 'multi'
    r.say(3000)
    await expect.poll(() => r.lines.length, { timeout: 5000 }).toBe(2)
    await r.stop()
    expect(r.lines).toEqual(['me:Hej då.', 'me:från whisper'])
    expect(r.made).toEqual(['whisper-server'])
  })

  it("a Pianissimo that cannot load does not silence the meeting: whisper reads the chunk", async () => {
    const r = await rig({ spoken: () => 'sv', ready: () => true, engine: async () => { throw new Error('onnxruntime would not load') } })
    r.say()
    await expect.poll(() => r.lines.length, { timeout: 5000 }).toBeGreaterThan(0)
    await r.stop()
    expect(r.lines[0]).toBe('me:från whisper')
  })

  it("gives Pianissimo the person's own words, the people in the meeting and their companies to prefer", async () => {
    setJargon(() => ['Hasselmus', 'NVI'])
    setOwnerName(() => 'Linnar Linnarsson')
    const r = await rig({ spoken: () => 'sv', ready: () => true, attendees: [{ name: 'Kalle Anka', email: 'kalle@ekologigruppen.se' }] as Meeting['attendees'] })
    r.say()
    await expect.poll(() => r.phrases.length, { timeout: 5000 }).toBeGreaterThan(0)
    await r.stop()
    const asked = r.phrases[0]!
    for (const w of ['Hasselmus', 'NVI', 'Kalle Anka', 'Kalle', 'Anka', 'Linnar Linnarsson', 'Ekologigruppen']) expect(asked).toContain(w)
    expect(asked[0]).toBe('Hasselmus')
  })
})

describe('boostTerms', () => {
  it('puts the own list first, then names whole and in parts, then companies, each once', () => {
    expect(boostTerms({ title: 't', names: ['Kalle Anka', 'Li Wu'], companies: ['Disney'], jargon: ['Hasselmus', 'kalle anka'] }))
      .toEqual(['Hasselmus', 'kalle anka', 'Li Wu', 'Kalle', 'Anka', 'Disney'])
  })
  it('leaves out parts of names shorter than three letters, and is empty for nothing', () => {
    expect(boostTerms({ title: '', names: ['Li Wu'], companies: [] })).toEqual(['Li Wu'])
    expect(boostTerms({ title: '', names: [], companies: [] })).toEqual([])
  })
  it('takes at most 300', () => {
    const jargon = Array.from({ length: 100 }, (_, i) => `ord${i}`)
    const names = Array.from({ length: 200 }, (_, i) => `Person${i} Efternamn${i}`)
    expect(boostTerms({ title: '', names, companies: [], jargon })).toHaveLength(300)
  })
  it('works from a meeting as the app asks for it', () => {
    const w = meetingWords({ title: 'Möte', titleSource: 'user', attendees: [{ name: 'Kalle Anka', email: 'kalle@disney.com' }] as Meeting['attendees'] })
    expect(boostTerms(w)).toContain('Disney')
  })
})
