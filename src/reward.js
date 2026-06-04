export class RewardModule {
  constructor(goal = 'wooden_pickaxe') {
    this.goal = goal;
    this.repeatCounts = new Map();
  }

  compute({ actionKey, deltaKK, error, inventory, predicted, observedState }) {
    const n = (this.repeatCounts.get(actionKey) || 0) + 1;
    this.repeatCounts.set(actionKey, n);

    const rRepeat = 0.2 / Math.sqrt(n);
    const rError = error ? -0.45 : 0;
    const rKK = deltaKK.length > 0 ? Math.min(0.6, 0.15 * deltaKK.length) : 0;
    const rGoal = inventory.some(i => i.name === this.goal) ? 4.0 : 0;
    const rPred = predicted ? this.predictionReward(predicted, observedState) : 0;

    const raw = rRepeat + rError + rKK + rGoal + rPred;
    return {
      total: Math.tanh(raw),
      parts: { rRepeat, rError, rKK, rGoal, rPred, raw }
    };
  }

  predictionReward(predicted, observed) {
    if (!predicted) return 0;
    let score = 0;
    if (predicted.error === observed.error) score += 0.12;
    const predictedDelta = new Set(predicted.deltaKK || []);
    const actualDelta = new Set(observed.deltaKK || []);
    let overlap = 0;
    for (const k of actualDelta) if (predictedDelta.has(k)) overlap += 1;
    if (actualDelta.size > 0) score += 0.18 * (overlap / actualDelta.size);
    return score;
  }
}
