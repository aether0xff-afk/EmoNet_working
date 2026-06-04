import fs from 'fs';

export function loadConfig() {
  const path = fs.existsSync('config.json') ? 'config.json' : 'config.example.json';
  return JSON.parse(fs.readFileSync(path, 'utf8'));
}
