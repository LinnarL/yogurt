import type { Transcribe, Line } from '../transcribe'
import { isGarbled } from '../transcribe'
import type { Engine } from './engine'

/**
 * The speech engine for Swedish: the same shape as the whisper transcriber (a chunk in, lines out), so the pipeline, the
 * echo check and the saved transcript do not know which engine read the audio. No language to detect or lock (the model
 * is Swedish), no prompt to read back (the person's words go to phrase boosting instead), and no speech detector in front
 * of it: the chunker has already cut the audio at pauses.
 */
export type PianissimoOpts = {
  engine: () => Promise<Engine>
  /** the words to boost: the person's own list, the people in the meeting and their companies, read for each chunk */
  phrases?: () => string[]
  /** how strongly the phrases are preferred (1: as KlangAI's pianissimo_onnx.boost) */
  alpha?: () => number
  /** the language a chunk was heard in, and how much was said (the notes are written in the main one) */
  onLanguage?: (track: Line['track'], language: string, chars: number, locked: boolean) => void
}

/** a sentence ends at a full stop, a question mark or an exclamation mark */
const SENTENCE_END = /[.?!]$/

export function createPianissimoTranscriber(o: PianissimoOpts): Transcribe {
  return async (c, opt) => {
    const engine = await o.engine()
    const r = await engine.recognize(c.samples, { phrases: o.phrases?.() ?? [], alpha: o.alpha?.() })
    if (!r.text.trim() || isGarbled(r.text)) return []
    // one line for each sentence, so a long chunk is not one long line
    const lines: Line[] = []
    let from = 0
    r.words.forEach((w, i) => {
      if (!SENTENCE_END.test(w.text) && i < r.words.length - 1) return
      const part = r.words.slice(from, i + 1)
      from = i + 1
      lines.push({ track: c.track, startMs: Math.round(c.startMs + part[0].startMs), endMs: Math.round(c.startMs + part[part.length - 1].endMs), text: part.map((x) => x.text).join(' '),
        words: part.map((x) => ({ text: x.text, startMs: Math.round(c.startMs + x.startMs), endMs: Math.round(c.startMs + x.endMs) })) })
    })
    if (!opt?.partial) o.onLanguage?.(c.track, 'swedish', r.text.length, true)
    return lines
  }
}
