export class KnowledgeStore {
  constructor() {
    this.kv = {
      visible_blocks: [],
      tree_blocks: [],
      stone_blocks: [],
      crafting_table_pos: [],
      inventory_items: [],
      craftable_items: [],
      failures: [],
      goals: []
    };
  }

  _keyOf(value) {
    if (value === null || value === undefined) return String(value);
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  add(kk, value) {
    if (!this.kv[kk]) this.kv[kk] = [];
    const key = this._keyOf(value);
    if (!this.kv[kk].some(v => this._keyOf(v) === key)) {
      this.kv[kk].push(value);
      return true;
    }
    return false;
  }

  observe(bot) {
    const delta = [];
    const blocks = bot.findBlocks({
      matching: b => b && b.name !== 'air',
      maxDistance: 12,
      count: 64
    });

    for (const pos of blocks) {
      const block = bot.blockAt(pos);
      if (!block) continue;
      if (this.add('visible_blocks', { name: block.name, x: pos.x, y: pos.y, z: pos.z })) delta.push('visible_blocks');
      if (block.name.includes('log') || block.name.includes('wood')) {
        if (this.add('tree_blocks', { name: block.name, x: pos.x, y: pos.y, z: pos.z })) delta.push('tree_blocks');
      }
      if (['stone', 'cobblestone', 'deepslate'].includes(block.name)) {
        if (this.add('stone_blocks', { name: block.name, x: pos.x, y: pos.y, z: pos.z })) delta.push('stone_blocks');
      }
      if (block.name === 'crafting_table') {
        if (this.add('crafting_table_pos', { x: pos.x, y: pos.y, z: pos.z })) delta.push('crafting_table_pos');
      }
    }

    const inventoryNames = bot.inventory.items().map(i => `${i.name}:${i.count}`);
    for (const item of inventoryNames) {
      if (this.add('inventory_items', item)) delta.push('inventory_items');
    }

    return [...new Set(delta)];
  }

  hasItem(name) {
    return this.kv.inventory_items.some(x => x.startsWith(`${name}:`));
  }

  countItemPrefix(prefix) {
    let total = 0;
    for (const x of this.kv.inventory_items) {
      const [name, count] = x.split(':');
      if (name.includes(prefix)) total += Number(count || 0);
    }
    return total;
  }

  snapshot() {
    return JSON.parse(JSON.stringify(this.kv));
  }
}
