"""Save WikiText-103 fitting prompts locally (jlens's own loader, fixed order)."""

import json
import sys
from pathlib import Path

from jlens.examples import load_wikitext_prompts

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
out = Path(__file__).resolve().parents[2] / "artifacts" / "jlens" / "data" / "wikitext_prompts.json"
prompts = load_wikitext_prompts(n)
out.write_text(json.dumps(prompts))
print(f"saved {len(prompts)} prompts to {out}")
