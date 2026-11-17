import type { Boost } from './boost'

/**
 * Greedy decoding of a token-and-duration transducer (the NeMo Parakeet TDT that Pianissimo is), as onnx-asr does it,
 * with the phrase bonus of KlangAI's pianissimo_onnx.boost added to the token choice.
 *
 * The decoder-joint network is asked once per step with the encoder frame at time t and the last token said. Its answer is
 * a score for each token (the blank last) and for each duration. A token other than the blank is written down; the duration
 * says how many frames to skip (0: stay and say another token, at most maxPerStep of them).
 */

export type JointOut = {
  /** a score for each of the vocabulary's tokens, the blank last */
  logits: Float32Array
  /** the duration chosen, in encoder frames */
  step: number
  /** the network's memory after this token, kept only when a token is written down */
  state: unknown
}
/** asks the decoder-joint network about the encoder frame t, after the token before (the blank at the start) */
export type Joint = (t: number, prevToken: number, state: unknown) => Promise<JointOut>

export type Decoded = { tokens: number[]; times: number[] }

export async function tdtDecode(o: {
  frames: number
  blank: number
  maxPerStep: number
  joint: Joint
  /** the network's memory before the first token */
  start: unknown
  boost?: { tree: Boost; alpha: number }
}): Promise<Decoded> {
  const tokens: number[] = [], times: number[] = []
  let state = o.start, boostState = 0, t = 0, emitted = 0
  while (t < o.frames) {
    const r = await o.joint(t, tokens.length ? tokens[tokens.length - 1] : o.blank, state)
    let token = argmax(r.logits, r.logits.length)
    if (token !== o.blank) {
      if (o.boost) {
        const { bonus, next } = o.boost.tree.scores(boostState)
        // the blank is not in the bonus: only a token that is said can be helped
        let best = -Infinity
        for (let i = 0; i < o.blank; i++) { const v = r.logits[i] + o.boost.alpha * bonus[i]; if (v > best) { best = v; token = i } }
        boostState = next[token]
      }
      state = r.state
      tokens.push(token)
      times.push(t)
      emitted++
    }
    if (r.step > 0) { t += r.step; emitted = 0 }
    else if (token === o.blank || emitted === o.maxPerStep) { t++; emitted = 0 }
  }
  return { tokens, times }
}

function argmax(a: Float32Array, n: number): number {
  let best = 0
  for (let i = 1; i < n; i++) if (a[i] > a[best]) best = i
  return best
}
