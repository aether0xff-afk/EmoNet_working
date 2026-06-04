function normalize(table) {
  const sum = Object.values(table).reduce((a, b) => a + b, 0) || 1;
  for (const k of Object.keys(table)) table[k] = Math.max(0.001, table[k] / sum);
  return table;
}

function sample(table) {
  const r = Math.random();
  let acc = 0;
  for (const [k, p] of Object.entries(table)) {
    acc += p;
    if (r <= acc) return k;
  }
  return Object.keys(table).at(-1);
}

export class ABCPolicy {
  constructor({ epsilon = 0.15, learningRate = 0.08 } = {}) {
    this.epsilon = epsilon;
    this.lr = learningRate;
    this.A = normalize({ observe: 1, explore: 1, mine: 1, craft: 1 });
    this.B = normalize({ nearest: 1, safe: 1, random: 1, known: 1 });
    this.C = normalize({ tree: 1, stone: 1, self: 1, front: 1, known_area: 1 });
  }

  sampleAction() {
    return { A: sample(this.A), B: sample(this.B), C: sample(this.C) };
  }

  candidates(n = 8) {
    const out = [];
    const seen = new Set();
    while (out.length < n && seen.size < 200) {
      const a = this.sampleAction();
      const key = this.key(a);
      seen.add(key);
      if (!out.some(x => this.key(x) === key)) out.push(a);
    }
    return out;
  }

  key(a) {
    return `${a.A}/${a.B}/${a.C}`;
  }

  update(action, reward, weight = 1) {
    const delta = this.lr * reward * weight;
    for (const [tableName, choice] of [['A', action.A], ['B', action.B], ['C', action.C]]) {
      const table = this[tableName];
      table[choice] = Math.max(0.001, table[choice] + delta);
      normalize(table);
    }
  }

  snapshot() {
    return { A: { ...this.A }, B: { ...this.B }, C: { ...this.C } };
  }
}
