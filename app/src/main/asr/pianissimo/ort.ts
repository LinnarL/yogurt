import type { Ort } from './engine'

/** onnxruntime-node, loaded when the first Swedish chunk (or the speed check) needs it: its native files are not read at start-up */
export const loadOrt = async (): Promise<Ort> => (await import('onnxruntime-node')) as unknown as Ort
