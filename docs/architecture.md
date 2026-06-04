# Architecture

This MVP transfers the paper architecture into a Minecraft environment.

```text
Minecraft observation
  -> KnowledgeStore (KK/KV)
  -> ABCPolicy (WHAT / HOW / WHERE)
  -> RewardModule
  -> ProphecyModule
  -> ImaginationCycle
  -> execute one action
  -> observe again
```

## Module mapping

| Paper prototype | Minecraft MVP |
| --- | --- |
| nmap XML observation | Mineflayer world and inventory observation |
| KK/KV store | blocks, items, crafting-table positions, failures |
| Policy A/B/C | WHAT, HOW, WHERE action factors |
| FLAG discovery | acquisition of the configured goal item |
| Prophecy | empirical next-state and reward prediction |
| Imagination Cycle | short candidate rollouts before execution |

## Current scope

The first milestone is intentionally small: acquire a `wooden_pickaxe` in a local controlled Minecraft Java server. The current Prophecy module is a lightweight empirical transition model rather than a Transformer. It is designed to validate the closed loop before introducing a trainable sequence model.

## Planned experiment conditions

- `C0`: random baseline
- `C1`: factorized A/B/C policy with extrinsic reward
- `C2`: C1 plus prediction-based intrinsic reward
- `C3`: C2 plus Imagination Cycle
