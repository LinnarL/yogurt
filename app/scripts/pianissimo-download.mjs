// Downloads the Swedish speech model (Pianissimo, ONNX) into a folder, as the app does: the files the catalog lists, pinned to
// a Hugging Face commit and checked against their sha256. For CI and for trying the engine without the app:
//   node app/scripts/pianissimo-download.mjs <folder>
// then PIANISSIMO_MODEL=<folder> pnpm -C app exec vitest run test/pianissimoEngine.test.ts
import { createHash } from 'node:crypto'
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'

const catalog = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../../catalog/catalog.json'), 'utf8'))
const model = catalog.speechSv
const out = process.argv[2]
if (!model) throw new Error('the catalog has no Swedish speech model')
if (!out) throw new Error('usage: node pianissimo-download.mjs <folder>')
mkdirSync(out, { recursive: true })

const sha256 = (path) => new Promise((resolve, reject) => {
  const h = createHash('sha256')
  import('node:fs').then(({ createReadStream }) => createReadStream(path).on('data', (c) => h.update(c)).on('error', reject).on('end', () => resolve(h.digest('hex'))))
})

for (const f of model.files) {
  // plain names only, as the app insists on
  if (!/^(?!\.)[\w.-]+$/.test(f.path)) throw new Error(`unsafe file name ${f.path}`)
  const path = join(out, f.path)
  if (existsSync(path) && (await sha256(path)) === f.sha256) { console.log(`${f.path}: already here`); continue }
  console.log(`${f.path}: downloading ${(f.bytes / 1e6).toFixed(1)} MB`)
  const res = await fetch(f.url)
  if (!res.ok || !res.body) throw new Error(`${f.path}: HTTP ${res.status}`)
  await pipeline(Readable.fromWeb(res.body), createWriteStream(path + '.part'))
  if ((await sha256(path + '.part')) !== f.sha256) { rmSync(path + '.part', { force: true }); throw new Error(`${f.path}: the download does not match its sha256`) }
  renameSync(path + '.part', path)
}
console.log(`${model.name}: ready in ${out}`)
