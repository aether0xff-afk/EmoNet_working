import fs from 'fs';
import path from 'path';

export class JsonlLogger {
  constructor() {
    fs.mkdirSync('logs', { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.file = path.join('logs', `run-${stamp}.jsonl`);
  }

  write(obj) {
    fs.appendFileSync(this.file, JSON.stringify(obj) + '\n');
  }
}
