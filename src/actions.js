import { goals } from 'mineflayer-pathfinder';
import { Vec3 } from 'vec3';

const { GoalNear } = goals;

function posObjToVec3(p) {
  return new Vec3(p.x, p.y, p.z);
}

async function goNear(bot, pos, range = 2) {
  await bot.pathfinder.goto(new GoalNear(pos.x, pos.y, pos.z, range));
}

async function explore(bot) {
  const yaw = Math.random() * Math.PI * 2;
  bot.look(yaw, 0, true);
  bot.setControlState('forward', true);
  if (Math.random() < 0.35) bot.setControlState('jump', true);
  await bot.waitForTicks(30 + Math.floor(Math.random() * 30));
  bot.clearControlStates();
}

async function mineNearest(bot, knowledge, kind) {
  const list = kind === 'stone' ? knowledge.kv.stone_blocks : knowledge.kv.tree_blocks;
  const target = list[0];
  if (!target) throw new Error(`no_known_${kind}`);
  const pos = posObjToVec3(target);
  const block = bot.blockAt(pos);
  if (!block) throw new Error(`missing_block_${kind}`);
  await goNear(bot, pos, 3);
  await bot.dig(block);
}

async function craftWoodenPickaxe(bot) {
  const mcData = (await import('minecraft-data')).default(bot.version);
  const logs = bot.inventory.items().filter(i => i.name.includes('log'));
  if (logs.length > 0) {
    const planksRecipe = bot.recipesFor(mcData.itemsByName.oak_planks?.id ?? mcData.itemsByName.spruce_planks?.id, null, 1, null)[0];
    if (planksRecipe) await bot.craft(planksRecipe, 1, null);
  }

  const tableItem = mcData.itemsByName.crafting_table;
  if (tableItem && !bot.inventory.items().some(i => i.name === 'crafting_table')) {
    const recipe = bot.recipesFor(tableItem.id, null, 1, null)[0];
    if (recipe) await bot.craft(recipe, 1, null);
  }

  let tableBlock = bot.findBlock({ matching: b => b.name === 'crafting_table', maxDistance: 5 });
  if (!tableBlock) {
    const table = bot.inventory.items().find(i => i.name === 'crafting_table');
    if (table) {
      const ref = bot.blockAt(bot.entity.position.offset(1, -1, 0));
      await bot.equip(table, 'hand');
      await bot.placeBlock(ref, new Vec3(0, 1, 0));
      tableBlock = bot.findBlock({ matching: b => b.name === 'crafting_table', maxDistance: 5 });
    }
  }

  const sticks = mcData.itemsByName.stick;
  if (sticks && !bot.inventory.items().some(i => i.name === 'stick')) {
    const recipe = bot.recipesFor(sticks.id, null, 1, tableBlock)[0] || bot.recipesFor(sticks.id, null, 1, null)[0];
    if (recipe) await bot.craft(recipe, 1, tableBlock);
  }

  const pickaxe = mcData.itemsByName.wooden_pickaxe;
  if (!pickaxe) throw new Error('no_wooden_pickaxe_item');
  const recipe = bot.recipesFor(pickaxe.id, null, 1, tableBlock)[0];
  if (!recipe) throw new Error('wooden_pickaxe_not_craftable');
  await bot.craft(recipe, 1, tableBlock);
}

export async function executeAction(bot, knowledge, action) {
  if (action.A === 'observe') {
    await bot.waitForTicks(10);
    return { ok: true };
  }

  if (action.A === 'explore') {
    await explore(bot);
    return { ok: true };
  }

  if (action.A === 'mine') {
    if (action.C === 'stone') await mineNearest(bot, knowledge, 'stone');
    else await mineNearest(bot, knowledge, 'tree');
    return { ok: true };
  }

  if (action.A === 'craft') {
    await craftWoodenPickaxe(bot);
    return { ok: true };
  }

  await bot.waitForTicks(5);
  return { ok: true };
}
