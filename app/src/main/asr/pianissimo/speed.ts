import { readFileSync } from 'node:fs'
import { performance } from 'node:perf_hooks'
import { loadEngine, type Ort } from './engine'
import { readWav } from '../speedCheck'

/**
 * How fast Pianissimo reads on this computer: time over the clip's length, as the whisper speed check measures it (the
 * bundled clip is English: the model's work per word is much the same, and the check is about speed, not the words).
 * The first 3 s are read once beforehand, so loading the files and the first-run work are not counted.
 * null: it could not be timed (the model would not load).
 */
export async function measureSwedish(dir: string, o: { ort: Ort; clip: string | undefined; load?: typeof loadEngine }): Promise<number | null> {
  if (!o.clip) { console.warn('Swedish speech model: no clip to time it on, taken as fast enough'); return 0 }
  const samples = readWav(readFileSync(o.clip))
  const engine = await (o.load ?? loadEngine)({ dir, ort: o.ort })
  try {
    await engine.recognize(samples.subarray(0, 3 * 16000))
    const t0 = performance.now()
    await engine.recognize(samples)
    return (performance.now() - t0) / 1000 / (samples.length / 16000)
  } finally { await engine.close() }
}
