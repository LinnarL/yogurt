/**
 * Phrase boosting for Pianissimo: a bonus for the tokens that continue one of the person's own words or phrases.
 * A TypeScript port of KlangAI's phrase_boost.py (numpy), itself a port of NeMo's GPU-PB boosting tree
 * (ContextGraph and GPUBoostingTreeModel, NeMo 2.7, Apache 2.0): a trie of the phrases' tokens with fail links
 * (Aho-Corasick), so a half-said phrase that goes wrong falls back to the longest suffix that is still a prefix.
 */

type Node = { id: number; children: Map<number, Node>; tokenScore: number; nodeScore: number; isEnd: boolean; fail: Node }

export type Boost = {
  /** the bonus for each token, and the state each token leads to, from this state (0 is the start) */
  scores(state: number): { bonus: Float32Array; next: Int32Array }
}

/**
 * @param phrases each already turned into tokens
 * @param vocab the tokens that can be boosted (the model's own, not the blank)
 */
export function createBoost(phrases: number[][], vocab: number, contextScore = 1, depthScaling = 2): Boost {
  const root = { id: 0, children: new Map(), tokenScore: 0, nodeScore: 0, isEnd: false } as unknown as Node
  root.fail = root
  const nodes: Node[] = [root]
  const seen = new Set<string>()
  for (const tokens of phrases) {
    const key = tokens.join(',')
    if (!tokens.length || seen.has(key)) continue
    seen.add(key)
    let node = root
    tokens.forEach((token, i) => {
      const last = i === tokens.length - 1
      let child = node.children.get(token)
      if (!child) {
        const tokenScore = i > 0 ? contextScore * depthScaling + Math.log(i + 1) : contextScore
        child = { id: nodes.length, children: new Map(), tokenScore, nodeScore: node.nodeScore + tokenScore, isEnd: last, fail: root }
        node.children.set(token, child)
        nodes.push(child)
      } else {
        child.tokenScore = Math.max(contextScore, child.tokenScore)
        child.nodeScore = node.nodeScore + child.tokenScore
        child.isEnd = last || child.isEnd
      }
      node = child
    })
  }
  // fail links, breadth first
  const queue: Node[] = []
  for (const child of root.children.values()) { child.fail = root; queue.push(child) }
  for (let q = 0; q < queue.length; q++) {
    const node = queue[q]
    for (const [token, child] of node.children) {
      let fail = node.fail
      while (!fail.children.has(token) && fail !== root) fail = fail.fail
      child.fail = fail.children.get(token) ?? root
      queue.push(child)
    }
  }

  const cache = new Map<number, { bonus: Float32Array; next: Int32Array }>()
  return {
    scores(state) {
      let hit = cache.get(state)
      if (hit) return hit
      const bonus = new Float32Array(vocab), next = new Int32Array(vocab).fill(-1)
      let node = nodes[state], takenBack = 0
      for (;;) {
        for (const [token, child] of node.children) {
          if (token < vocab && next[token] < 0) { bonus[token] = takenBack + child.tokenScore; next[token] = child.id }
        }
        if (node === root) {
          for (let t = 0; t < vocab; t++) if (next[t] < 0) { bonus[t] = takenBack; next[t] = 0 }
          break
        }
        if (!node.isEnd) takenBack += node.fail.nodeScore - node.nodeScore
        node = node.fail
      }
      hit = { bonus, next }
      cache.set(state, hit)
      return hit
    },
  }
}
