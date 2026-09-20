# Ouro/RLTT Components

This directory is the local remote-code/model-definition mirror for Ouro/RLTT.
The active weights are not here; they live in `shared/models/ouro_rltt_local/`.

## Model Definition

- **OuroConfig**: configuration class for loading the local model definition
  through Hugging Face `trust_remote_code`.
- **OuroModel / OuroForCausalLM**: looped transformer implementation used by
  the local RLTT checkpoint.
- **UniversalTransformerCache**: custom cache object used by the model. The
  local runtime defaults `use_cache=False` because the cache setter path is
  fragile in the current wrapper.
- **Adaptive compute hooks**: inference-only controls for loop/exit behavior.
  They are available for calibration but are not the default runtime mode.

## Tokenizer And Chat Assets

- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `special_tokens_map.json`: tokenizer assets.
- `chat_template.jinja`: local chat template.

## Active Weight Location

- `shared/models/ouro_rltt_local/`: converted safetensors checkpoint used by
  Hunter-Seeker probes and the local-agent wrapper.
- `src/ouro_rltt/` itself should be treated as model code/config assets, not
  the active weight store.
