import { loadEngine, type Engine, type Ort } from './engine'

/** How long Pianissimo stays in memory after the last chunk: back-to-back meetings skip the load, like whisper-server's wait */
export const ENGINE_IDLE_MS = 10 * 60_000

/**
 * One Pianissimo engine while meetings need it: loaded when the first chunk comes (a few seconds), let go a while after
 * the last. The folder is asked for each time, so a model that was removed in Settings is not kept alive.
 */
export function createEngineHolder(o: { ort: () => Promise<Ort>; dir: () => string | undefined; idleMs?: number; load?: typeof loadEngine }) {
  let engine: Promise<Engine> | undefined, at: string | undefined, timer: NodeJS.Timeout | undefined
  const load = o.load ?? loadEngine

  const drop = () => {
    clearTimeout(timer)
    const e = engine
    engine = undefined; at = undefined
    void e?.then((x) => x.close()).catch(() => undefined)
  }
  return {
    /** the engine, loading it if need be; rejects when no model is installed or it cannot be loaded (then asked for again next time) */
    async get(): Promise<Engine> {
      const dir = o.dir()
      if (!dir) { drop(); throw new Error('the Swedish speech model is not installed') }
      if (engine && at !== dir) drop()
      if (!engine) {
        at = dir
        engine = o.ort().then((ort) => load({ dir, ort })).catch((e) => { engine = undefined; at = undefined; throw e })
      }
      const ready = await engine
      clearTimeout(timer)
      timer = setTimeout(drop, o.idleMs ?? ENGINE_IDLE_MS)
      timer.unref?.()
      return ready
    },
    /** the app quits, or the model is removed */
    stop: drop,
    loaded: () => !!engine,
  }
}
export type EngineHolder = ReturnType<typeof createEngineHolder>
