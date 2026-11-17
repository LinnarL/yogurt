/**
 * A reader and encoder for a SentencePiece BPE model (tokenizer.model), enough to turn a custom word into the tokens
 * Pianissimo's phrase boosting needs. Not a general SentencePiece: BPE only, no sampling, no unigram.
 * Checked against the python sentencepiece package (test/pianissimo.test.ts, scripts' fixture).
 */

export type Piece = { text: string; score: number; type: number }
/** SentencePiece's piece types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte */
const NORMAL = 1, UNKNOWN = 2, USER = 4, BYTE = 6
const SPACE = '▁'

/** the pieces of a model file, in id order, and whether unknown characters fall back to bytes */
export function readModel(buf: Uint8Array): { pieces: Piece[]; byteFallback: boolean; addDummyPrefix: boolean; removeExtraWhitespaces: boolean } {
  const pieces: Piece[] = []
  let byteFallback = false, addDummyPrefix = true, removeExtraWhitespaces = true
  for (const f of fields(buf)) {
    if (f.no === 1 && f.bytes) {
      let text = '', score = 0, type = NORMAL
      for (const g of fields(f.bytes)) {
        if (g.no === 1 && g.bytes) text = new TextDecoder().decode(g.bytes)
        else if (g.no === 2 && g.bytes) score = new DataView(g.bytes.buffer, g.bytes.byteOffset, 4).getFloat32(0, true)
        else if (g.no === 3 && g.int !== undefined) type = g.int
      }
      pieces.push({ text, score, type })
    } else if (f.no === 2 && f.bytes) {
      for (const g of fields(f.bytes)) if (g.no === 35 && g.int !== undefined) byteFallback = g.int === 1   // trainer_spec.byte_fallback
    } else if (f.no === 3 && f.bytes) {
      for (const g of fields(f.bytes)) {
        if (g.no === 3 && g.int !== undefined) addDummyPrefix = g.int === 1   // normalizer_spec.add_dummy_prefix
        else if (g.no === 4 && g.int !== undefined) removeExtraWhitespaces = g.int === 1   // normalizer_spec.remove_extra_whitespaces
      }
    }
  }
  if (!pieces.length) throw new Error('tokenizer model has no pieces')
  return { pieces, byteFallback, addDummyPrefix, removeExtraWhitespaces }
}

type Field = { no: number; int?: number; bytes?: Uint8Array }
/** the fields of one protobuf message (varints, length-delimited, fixed 32 and 64 bit) */
function* fields(buf: Uint8Array): Generator<Field> {
  let i = 0
  const varint = (): number => {
    let r = 0, shift = 1
    for (;;) {
      const b = buf[i++]
      if (b === undefined) throw new Error('truncated protobuf')
      r += (b & 0x7f) * shift
      if (!(b & 0x80)) return r
      shift *= 128
    }
  }
  while (i < buf.length) {
    const tag = varint(), no = Math.floor(tag / 8), wire = tag % 8
    if (wire === 0) yield { no, int: varint() }
    else if (wire === 2) { const n = varint(); yield { no, bytes: buf.subarray(i, i + n) }; i += n }
    else if (wire === 5) { yield { no, bytes: buf.subarray(i, i + 4) }; i += 4 }
    else if (wire === 1) { yield { no, bytes: buf.subarray(i, i + 8) }; i += 8 }
    else throw new Error(`unsupported protobuf wire type ${wire}`)
  }
}

export type Tokenizer = { encode(text: string): number[]; size: number }

/** BPE as SentencePiece does it: start from characters, then merge the adjacent pair whose joined piece scores highest */
export function createTokenizer(buf: Uint8Array): Tokenizer {
  const { pieces, byteFallback, addDummyPrefix, removeExtraWhitespaces } = readModel(buf)
  const id = new Map<string, number>()
  pieces.forEach((p, i) => { if ((p.type === NORMAL || p.type === USER) && !id.has(p.text)) id.set(p.text, i) })
  const unk = Math.max(0, pieces.findIndex((p) => p.type === UNKNOWN))
  const byteId = (b: number) => id.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`) ?? pieces.findIndex((p) => p.type === BYTE && p.text === `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`)

  function encode(text: string): number[] {
    // nmt_nfkc, spaces written as the lower block, a dummy space in front; extra whitespace removed only if the model says so
    let s = text.normalize('NFKC')
    if (removeExtraWhitespaces) s = s.replace(/\s+/g, ' ').trim()
    if (!s) return []
    s = (addDummyPrefix ? ' ' + s : s).replaceAll(' ', SPACE)
    let syms = [...s]
    for (;;) {
      let best = -1, bestScore = -Infinity
      for (let i = 0; i + 1 < syms.length; i++) {
        const p = id.get(syms[i] + syms[i + 1])
        if (p !== undefined && pieces[p].score > bestScore) { best = i; bestScore = pieces[p].score }
      }
      if (best < 0) break
      syms = [...syms.slice(0, best), syms[best] + syms[best + 1], ...syms.slice(best + 2)]
    }
    const out: number[] = []
    for (const sym of syms) {
      const p = id.get(sym)
      if (p !== undefined) out.push(p)
      else if (byteFallback) for (const b of new TextEncoder().encode(sym)) out.push(byteId(b) >= 0 ? byteId(b) : unk)
      else out.push(unk)
    }
    return out
  }
  return { encode, size: pieces.length }
}
