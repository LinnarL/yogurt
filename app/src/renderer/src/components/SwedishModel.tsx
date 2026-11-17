import { useEffect, useState } from 'react'
import type { SwedishState } from '../../../shared/types'
import { t } from '../../../shared/i18n'

/**
 * Settings → Language: the Swedish speech model (Pianissimo), an extra download. With it, Swedish meetings are read by a
 * model made for Swedish that also takes the person's own words (Jargon) as help; without it, whisper reads Swedish.
 */
export function SwedishModelSetting() {
  const [s, setS] = useState<SwedishState>()
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let current = false, active = true
    const off = window.yogurt.swedish.onState((state) => { current = true; setS(state) })
    void window.yogurt.swedish.state().then((state) => { if (active && !current) setS(state) })
    return () => { active = false; off() }
  }, [])
  // a catalog without the model: nothing to offer
  if (!s?.bytes) return null
  const working = s.status === 'downloading' || s.status === 'checking'
  const act = async () => {
    setBusy(true)
    try { if (s.status === 'ready' || s.status === 'downloading') await window.yogurt.swedish.remove(); else await window.yogurt.swedish.download() } finally { setBusy(false) }
  }
  const detail = s.status === 'downloading' ? `${Math.floor(s.fraction * 100)}%`
    : s.status === 'checking' ? t('Preparing…')
    : s.status === 'too-slow' ? t('This computer is too slow for the Swedish speech model.')
    : s.status === 'failed' ? (s.error ?? '')
    : t('Pianissimo reads Swedish more accurately, and spells your own words the way you typed them. It runs on this computer.')
  const label = s.status === 'ready' || s.status === 'downloading' ? (busy ? t('Removing…') : t('Remove'))
    : s.status === 'checking' ? t('Preparing…')
    : s.status === 'too-slow' || s.status === 'failed' ? t('Try again')
    : t('Download ({size} GB)', { size: (s.bytes / 1e9).toFixed(1) })
  return <div className="settings-row" data-testid="settings-swedish-model" data-status={s.status}>
    <div className="settings-row-copy">
      <p className="settings-label">{t('Swedish speech model')}</p>
      <p className="settings-description">{detail}</p>
    </div>
    <button data-testid="settings-swedish-model-action" className="workspace-button settings-action" aria-busy={busy || s.status === 'checking'} disabled={busy || s.status === 'checking'} onClick={() => void act()}>
      {label}
    </button>
  </div>
}
