import { mkdirSync, mkdtempSync, readFileSync, existsSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { CatalogSpeechSv } from '@yogurt/picker'
import { SwedishModel } from '../src/main/asr/pianissimo/model'
import { createEngineHolder } from '../src/main/asr/pianissimo/holder'
import type { Engine } from '../src/main/asr/pianissimo/engine'
import { friendly } from '../src/main/setup'
import type { DownloadDeps } from '../src/main/downloader'
import type { SwedishState } from '../src/shared/types'

const catalog: CatalogSpeechSv = {
  id: 'test-sv', name: 'Test (Swedish)', license: 'CC-BY-4.0', quant: 'int8', sizeGB: 0.001, bytes: 30, minMemGB: 0,
  files: [
    { path: 'a.onnx', url: 'https://example.test/a.onnx', sha256: 'sha-a', bytes: 10 },
    { path: 'b.txt', url: 'https://example.test/b.txt', sha256: 'sha-b', bytes: 20 },
  ],
}

let root: string, dir: string
beforeEach(() => { root = mkdtempSync(join(tmpdir(), 'sv-model-')); dir = join(root, 'asr-sv') })
afterEach(() => rmSync(root, { recursive: true, force: true }))

/** a download that writes the file; what it hashes to is what the catalog says unless a file is made to come out wrong */
function deps(o: { corrupt?: string; failTimes?: number; freeGB?: number } = {}) {
  let failures = o.failTimes ?? 0
  const downloaded: string[] = []
  const d: DownloadDeps = {
    async download(_url, to, fileName, onProgress) {
      if (failures-- > 0) throw new Error('fetch failed')
      onProgress({ fraction: 0.5, etaSec: 1 }); onProgress({ fraction: 1, etaSec: 0 })
      mkdirSync(to, { recursive: true })
      writeFileSync(join(to, fileName), fileName === o.corrupt ? 'bad' : 'ok')
      downloaded.push(fileName)
    },
    async hashFile(path) { return readFileSync(path, 'utf8') === 'bad' ? 'wrong' : path.endsWith('a.onnx') ? 'sha-a' : 'sha-b' },
    async freeDiskGB() { return o.freeGB ?? 1000 },
  }
  return { d, downloaded }
}
const make = (o: { download?: DownloadDeps; rtf?: () => Promise<number | null>; model?: CatalogSpeechSv | undefined } = {}) => {
  const measured: string[] = []
  const m = new SwedishModel({ dir, model: () => ('model' in o ? o.model : catalog), download: o.download ?? deps().d, friendly, retryDelaysMs: [1, 1],
    measure: async (d) => { measured.push(d); return o.rtf ? o.rtf() : 0.1 } })
  const seen: SwedishState['status'][] = []
  m.onState((s) => { if (seen.at(-1) !== s.status) seen.push(s.status) })
  return { m, measured, seen }
}

describe('the Swedish speech model', () => {
  it('is absent until it is downloaded, then downloaded, timed once and ready', async () => {
    const { d, downloaded } = deps()
    const { m, measured, seen } = make({ download: d })
    expect(m.state).toMatchObject({ status: 'absent', bytes: 30 })
    expect(m.path()).toBeUndefined()
    await m.start()
    expect(downloaded).toEqual(['a.onnx', 'b.txt'])
    expect(measured).toEqual([dir])
    expect(seen).toEqual(['downloading', 'checking', 'ready'])
    expect(m.state).toMatchObject({ status: 'ready', fraction: 1 })
    expect(m.path()).toBe(dir)
  })

  it('is ready at once after a restart, without downloading or timing again', async () => {
    await make().m.start()
    const { d, downloaded } = deps()
    const again = make({ download: d })
    expect(again.m.state.status).toBe('ready')
    await again.m.start()
    expect(downloaded).toEqual([])
    expect(again.measured).toEqual([])
  })

  it('reports the progress over all the files, forward only', async () => {
    const { m } = make()
    const fractions: number[] = []
    m.onState((s) => { if (s.status === 'downloading') fractions.push(s.fraction) })
    await m.start()
    expect(fractions.length).toBeGreaterThan(2)
    expect(fractions.every((f, i) => i === 0 || f >= fractions[i - 1])).toBe(true)
    expect(Math.max(...fractions)).toBeCloseTo(1)
  })

  it('too slow on this computer: the files go again, and it stays known across a restart without downloading', async () => {
    const { m } = make({ rtf: async () => 0.9 })
    await m.start()
    expect(m.state.status).toBe('too-slow')
    expect(m.path()).toBeUndefined()
    expect(existsSync(join(dir, 'a.onnx'))).toBe(false)
    const { d, downloaded } = deps()
    const again = make({ download: d })
    expect(again.m.state.status).toBe('too-slow')
    // asking again times it again (downloading the files)
    await again.m.start()
    expect(downloaded).toEqual(['a.onnx', 'b.txt'])
    expect(again.measured).toHaveLength(1)
  })

  it('a model that cannot be timed counts as too slow', async () => {
    const { m } = make({ rtf: async () => { throw new Error('would not load') } })
    await m.start()
    expect(m.state.status).toBe('too-slow')
  })

  it('a download that fails is tried again, then says what went wrong, and can be started again', async () => {
    const { d } = deps({ failTimes: 5 })
    const { m } = make({ download: d })
    await m.start()
    expect(m.state).toMatchObject({ status: 'failed', error: 'Download interrupted. Check your connection and try again.' })
    expect(m.path()).toBeUndefined()
    const ok = deps()
    const again = make({ download: ok.d })
    await again.m.start()
    expect(again.m.state.status).toBe('ready')
  })

  it('a download that comes out wrong twice is a damaged file, and nothing is timed', async () => {
    const { d } = deps({ corrupt: 'b.txt' })
    const { m, measured } = make({ download: d })
    await m.start()
    expect(m.state).toMatchObject({ status: 'failed', error: 'The downloaded file is damaged.' })
    expect(measured).toEqual([])
  })

  it('does not start without room for it on the disk', async () => {
    const { m } = make({ download: deps({ freeGB: 1 }).d })
    await m.start()
    expect(m.state.status).toBe('failed')
    expect(m.state.error).toMatch(/GB more free space/)
  })

  it('removes its files and what was found about them', async () => {
    const { m } = make()
    await m.start()
    await m.remove()
    expect(m.state.status).toBe('absent')
    expect(existsSync(dir)).toBe(false)
    expect(make().m.state.status).toBe('absent')
  })

  it('stops a download that is on its way when it is removed', async () => {
    let release!: () => void
    const gate = new Promise<void>((r) => { release = r })
    const slow: DownloadDeps = { ...deps().d, async download(_u, to, f, _p, signal) { await gate; signal?.throwIfAborted(); mkdirSync(to, { recursive: true }); writeFileSync(join(to, f), 'ok') } }
    const { m, measured } = make({ download: slow })
    const started = m.start()
    await new Promise((r) => setTimeout(r, 10))
    const removed = m.remove()
    release()
    await Promise.all([started, removed])
    expect(m.state.status).toBe('absent')
    expect(measured).toEqual([])
  })

  it('starting twice downloads once', async () => {
    const { d, downloaded } = deps()
    const { m } = make({ download: d })
    await Promise.all([m.start(), m.start()])
    expect(downloaded).toEqual(['a.onnx', 'b.txt'])
  })

  it('offers nothing when the catalog has no Swedish model', async () => {
    const { m } = make({ model: undefined })
    expect(m.state).toMatchObject({ status: 'absent', bytes: 0 })
    await m.start()
    expect(m.state.status).toBe('absent')
  })
})

describe('the engine holder', () => {
  const fakeEngine = () => { const e = { closed: false, recognize: async () => ({ text: '', words: [] }), close: async () => { e.closed = true } }; return e as Engine & { closed: boolean } }
  const ort = async () => ({}) as never
  const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

  it('loads once for many asks, and lets go after the wait', async () => {
    const made: Array<Engine & { closed: boolean }> = []
    const h = createEngineHolder({ ort, dir: () => '/m', idleMs: 40, load: async () => { const e = fakeEngine(); made.push(e); return e } })
    const [a, b] = await Promise.all([h.get(), h.get()])
    expect(a).toBe(b)
    expect(made).toHaveLength(1)
    await wait(80)
    expect(made[0].closed).toBe(true)
    expect(h.loaded()).toBe(false)
    await h.get()
    expect(made).toHaveLength(2)
    h.stop()
  })

  it('keeps going while it is used', async () => {
    const made: Array<Engine & { closed: boolean }> = []
    const h = createEngineHolder({ ort, dir: () => '/m', idleMs: 60, load: async () => { const e = fakeEngine(); made.push(e); return e } })
    for (let i = 0; i < 4; i++) { await h.get(); await wait(30) }
    expect(made[0].closed).toBe(false)
    h.stop()
  })

  it('has nothing when no model is installed, and drops the engine of a model that went', async () => {
    let at: string | undefined = '/m'
    const made: Array<Engine & { closed: boolean }> = []
    const h = createEngineHolder({ ort, dir: () => at, load: async () => { const e = fakeEngine(); made.push(e); return e } })
    await h.get()
    at = undefined
    await expect(h.get()).rejects.toThrow(/not installed/)
    await wait(5)
    expect(made[0].closed).toBe(true)
  })

  it('asks again after a failed load, and loads again for another folder', async () => {
    let fail = true, at = '/a'
    const made: string[] = []
    const h = createEngineHolder({ ort, dir: () => at, load: async (o) => { if (fail) throw new Error('no'); made.push(o.dir); return fakeEngine() } })
    await expect(h.get()).rejects.toThrow('no')
    fail = false
    await h.get()
    at = '/b'
    await h.get()
    expect(made).toEqual(['/a', '/b'])
    h.stop()
  })
})
