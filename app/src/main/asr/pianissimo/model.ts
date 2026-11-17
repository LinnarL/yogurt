import type { CatalogSpeechSv } from '@yogurt/picker'
import { KEEPS_UP_RTF } from '@yogurt/picker'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import type { SwedishState } from '../../../shared/types'
import { ChecksumError, DiskSpaceError, ensureFile, isVerified, safeRelPath, type DownloadDeps } from '../../downloader'

/**
 * The Swedish speech model's files on this computer: an extra download picked in Settings → Language, kept in its own
 * folder next to the whisper models. Downloaded file by file (each checked against the catalog's hash), then timed once on
 * the bundled clip: a computer that cannot read it twice as fast as it plays does not use it, and the files go again.
 */
type Deps = {
  /** where the model's folder goes */
  dir: string
  model: () => CatalogSpeechSv | undefined
  download: DownloadDeps
  /** fake mode only: treat every file as having this hash */
  sha256Override?: string
  /** how long one reading of the speed clip took over its length (null: it could not be timed or ran out of time) */
  measure: (dir: string) => Promise<number | null>
  /** the friendly text for a failed download (setup.ts's) */
  friendly: (err: unknown) => string
  retryDelaysMs?: number[]
}

/** what the speed check found, kept so a restart does not time it again */
const VERDICT = 'speed.json'
/** room left on the disk after the download, GB */
const HEADROOM_GB = 2

export class SwedishModel {
  private _state: SwedishState = { status: 'absent', fraction: 0, bytes: 0 }
  private listeners = new Set<(s: SwedishState) => void>()
  private running: Promise<void> | undefined
  private abort: AbortController | undefined

  constructor(private d: Deps) { this.refresh() }

  get state(): SwedishState { return this._state }
  onState(cb: (s: SwedishState) => void) { this.listeners.add(cb); return () => { this.listeners.delete(cb) } }
  private set(s: Partial<SwedishState>) {
    this._state = { ...this._state, ...s }
    for (const l of this.listeners) { try { l(this._state) } catch (e) { console.error('Swedish model listener failed', e) } }
  }

  /** the folder to read the model from; undefined until it is downloaded, checked and fast enough */
  path(): string | undefined { return this._state.status === 'ready' ? this.d.dir : undefined }

  private sha(sha: string) { return this.d.sha256Override ?? sha }
  private verdict(): { id: string; rtf: number | null } | undefined {
    try { return JSON.parse(readFileSync(join(this.d.dir, VERDICT), 'utf8')) } catch { return undefined }
  }
  private complete(m: CatalogSpeechSv) { return m.files.every((f) => isVerified(this.d.dir, safeRelPath(f.path), this.sha(f.sha256))) }

  /** Read what is on the disk: a complete model that was timed fast enough is ready (also after a restart) */
  refresh() {
    const m = this.d.model()
    if (!m) return this.set({ status: 'absent', fraction: 0, bytes: 0 })
    const v = this.verdict(), fast = v?.id === m.id && v.rtf !== null && v.rtf <= KEEPS_UP_RTF
    // too slow stays known after the files are deleted, so nothing downloads again by itself
    const status = v?.id === m.id && !fast ? 'too-slow' : fast && this.complete(m) ? 'ready' : 'absent'
    this.set({ status, fraction: status === 'ready' ? 1 : 0, bytes: m.bytes, error: undefined })
  }

  /** Download what is missing and time it. Safe to call again while it runs. */
  start(): Promise<void> {
    if (this.running) return this.running
    const m = this.d.model()
    if (!m || this._state.status === 'ready') return Promise.resolve()
    // asked again after "too slow" (another computer, or a quieter moment): time it again
    if (this._state.status === 'too-slow') rmSync(join(this.d.dir, VERDICT), { force: true })
    this.abort = new AbortController()
    this.running = this.run(m, this.abort.signal).finally(() => { this.running = undefined; this.abort = undefined })
    return this.running
  }

  /** Stop a download and delete the files (and what was found about them) */
  async remove(): Promise<void> {
    this.abort?.abort()
    await this.running?.catch(() => undefined)
    rmSync(this.d.dir, { recursive: true, force: true })
    this.refresh()
  }

  private async run(m: CatalogSpeechSv, signal: AbortSignal) {
    try {
      mkdirSync(this.d.dir, { recursive: true })
      if (!this.complete(m)) {
        const missing = m.files.filter((f) => !isVerified(this.d.dir, safeRelPath(f.path), this.sha(f.sha256)))
        const need = missing.reduce((n, f) => n + f.bytes / 1e9, 0) + HEADROOM_GB - (await this.d.download.freeDiskGB(this.d.dir))
        if (need > 0) throw new DiskSpaceError(Math.ceil(need))
        this.set({ status: 'downloading', fraction: 0, bytes: m.bytes, error: undefined })
        const total = m.files.reduce((n, f) => n + f.bytes, 0)
        let done = 0
        for (const f of m.files) {
          await this.fetch(m, f, (fraction) => this.set({ status: 'downloading', fraction: Math.min(1, (done + f.bytes * fraction) / total) }), signal)
          done += f.bytes
        }
      }
      // the files are all here: time them, unless that was done for this very model before
      if (this.verdict()?.id !== m.id) {
        this.set({ status: 'checking', fraction: 1 })
        const rtf = await this.d.measure(this.d.dir).catch((err) => { console.warn('timing the Swedish speech model failed', err); return null })
        signal.throwIfAborted()
        writeFileSync(join(this.d.dir, VERDICT), JSON.stringify({ id: m.id, rtf }))
        console.log(`Swedish speech model: ${rtf === null ? 'could not be timed' : `${rtf.toFixed(2)} x the clip's length`}`)
      }
      this.refresh()
      // too slow here: not worth the disk
      if (this._state.status === 'too-slow') for (const f of m.files) rmSync(join(this.d.dir, f.path), { force: true })
    } catch (err) {
      if (signal.aborted) return
      console.error('the Swedish speech model did not download', err)
      this.set({ status: 'failed', error: this.d.friendly(err) })
    }
  }

  private async fetch(m: CatalogSpeechSv, f: CatalogSpeechSv['files'][number], onProgress: (fraction: number) => void, signal: AbortSignal) {
    const delays = this.d.retryDelaysMs ?? [2000, 8000]
    for (let attempt = 0; ; attempt++) {
      try {
        signal.throwIfAborted()
        return await ensureFile({ url: f.url, sha256: this.sha(f.sha256), sizeGB: f.bytes / 1e9, dir: this.d.dir, fileName: safeRelPath(f.path), onProgress: (p) => onProgress(p.fraction), signal }, this.d.download)
      } catch (err) {
        signal.throwIfAborted()
        if (err instanceof DiskSpaceError || err instanceof ChecksumError || attempt >= delays.length) throw err
        await delay(delays[attempt], undefined, { signal })
      }
    }
  }
}
