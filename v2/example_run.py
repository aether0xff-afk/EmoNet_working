from emonet import EmotionArchitecture, default_config

cfg = default_config()
model = EmotionArchitecture(cfg)
out = model.infer("왜 이렇게 일이 많지", latent_dim=64)
print(out.prompt["prompt"])
print(out.h_t)
print(out.s[:8])
