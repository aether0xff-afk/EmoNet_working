import mineflayer from 'mineflayer';
import { pathfinder, Movements } from 'mineflayer-pathfinder';
import mcDataLoader from 'minecraft-data';
import { loadConfig } from './config.js';
import { KnowledgeStore } from './knowledge.js';
import { ABCPolicy } from './policy.js';
import { RewardModule } from './reward.js';
import { ProphecyModule } from './prophecy.js';
import { ImaginationCycle } from './imagination.js';
import { executeAction } from './actions.js';
import { JsonlLogger } from './logger.js';

const config = loadConfig();
const bot = mineflayer.createBot({
  host: config.host,
  port: config.port,
  username: config.username,
  version: config.version || false
});

bot.loadPlugin(pathfinder);

const knowledge = new KnowledgeStore();
const policy = new ABCPolicy(config.policy);
const prophecy = new ProphecyModule();
const rewarder = new RewardModule(config.goal);
const imagination = new ImaginationCycle(config.imagination);
const logger = new JsonlLogger();

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function inventorySummary() {
  return bot.inventory.items().map(i => ({ name: i.name, count: i.count }));
}

function hasGoal() {
  return bot.inventory.items().some(i => i.name === config.goal);
}

async function runEpisode() {
  for (let step = 0; step < config.episodeSteps; step++) {
    const beforeDelta = knowledge.observe(bot);
    const chooseResult = config.imagination.enabled
      ? imagination.choose(policy, prophecy)
      : { selected: policy.sampleAction(), scored: [] };

    const action = chooseResult.selected;
    const actionKey = policy.key(action);
    const predicted = prophecy.predict(action);
    let error = null;

    try {
      await executeAction(bot, knowledge, action);
    } catch (e) {
      error = e.message || String(e);
      knowledge.add('failures', { actionKey, error });
    }

    await sleep(250);
    const deltaKK = [...new Set([...beforeDelta, ...knowledge.observe(bot)])];
    const observedState = { deltaKK, error: Boolean(error) };
    const reward = rewarder.compute({
      actionKey,
      deltaKK,
      error,
      inventory: inventorySummary(),
      predicted,
      observedState
    });

    policy.update(action, reward.total);
    prophecy.update(action, observedState, reward.total);

    const row = {
      step,
      action,
      actionKey,
      predicted,
      observedState,
      reward,
      inventory: inventorySummary(),
      policy: policy.snapshot(),
      imagination: chooseResult.scored.slice(0, 3),
      goalFound: hasGoal(),
      time: new Date().toISOString()
    };
    logger.write(row);
    console.log(`[${step}] ${actionKey} reward=${reward.total.toFixed(3)} delta=${deltaKK.join(',') || '-'} error=${error || '-'} goal=${row.goalFound}`);

    if (row.goalFound) {
      console.log(`GOAL FOUND: ${config.goal}`);
      break;
    }
  }
  bot.quit();
}

bot.once('spawn', async () => {
  const mcData = mcDataLoader(bot.version);
  bot.pathfinder.setMovements(new Movements(bot, mcData));
  console.log('Bot spawned. Starting RL loop...');
  await runEpisode();
});

bot.on('error', err => console.error('Bot error:', err));
bot.on('kicked', reason => console.error('Kicked:', reason));
