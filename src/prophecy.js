export class ProphecyModule {
  constructor() {
    this.stats = new Map();
  }

  key(action) {
    return `${action.A}/${action.B}/${action.C}`;
  }

  predict(action) {
    const key = this.key(action);
    const row = this.stats.get(key);
    if (!row || row.count === 0) {
      return { deltaKK: [], error: false, expectedReward: 0, confidence: 0 };
    }
    const deltaKK = Object.entries(row.deltaCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k]) => k);
    return {
      deltaKK,
      error: row.errorCount / row.count > 0.5,
      expectedReward: row.rewardSum / row.count,
      confidence: Math.min(1, row.count / 10)
    };
  }

  update(action, observedState, reward) {
    const key = this.key(action);
    if (!this.stats.has(key)) {
      this.stats.set(key, { count: 0, errorCount: 0, rewardSum: 0, deltaCounts: {} });
    }
    const row = this.stats.get(key);
    row.count += 1;
    row.rewardSum += reward;
    if (observedState.error) row.errorCount += 1;
    for (const kk of observedState.deltaKK || []) {
      row.deltaCounts[kk] = (row.deltaCounts[kk] || 0) + 1;
    }
  }

  snapshot() {
    return Object.fromEntries(this.stats.entries());
  }
}
