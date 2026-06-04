export class ImaginationCycle {
  constructor({ candidates = 8, horizon = 3, gamma = 0.72 } = {}) {
    this.candidates = candidates;
    this.horizon = horizon;
    this.gamma = gamma;
  }

  choose(policy, prophecy) {
    const candidates = policy.candidates(this.candidates);
    const scored = candidates.map(action => {
      let score = 0;
      let discount = 1;
      let current = action;
      const rollout = [];

      for (let t = 0; t < this.horizon; t++) {
        const pred = prophecy.predict(current);
        const novelty = (pred.deltaKK || []).length * 0.12;
        const uncertaintyBonus = (1 - pred.confidence) * 0.05;
        const errorPenalty = pred.error ? -0.2 : 0;
        const s = pred.expectedReward + novelty + uncertaintyBonus + errorPenalty;
        score += discount * s;
        rollout.push({ action: current, pred, score: s });
        discount *= this.gamma;
        current = policy.sampleAction();
      }
      return { action, score, rollout };
    });

    scored.sort((a, b) => b.score - a.score);
    return { selected: scored[0].action, scored };
  }
}
