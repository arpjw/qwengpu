"""Fixed lane commit commands and model-derived memory write inputs."""

from chipformer.dsl.graph import Expr

COMMIT_BASE = 2688
COMMIT_COUNT = 8
INTERNAL_STORES_LAYOUT = "qwengpu-resident-model-stores-9b-v1"


def key_matches(c, bits, base, enabled):
    return sum(
        (value if base >> i & 1 else c.one - value for i, value in enumerate(bits)),
        Expr(),
    ) + enabled - len(bits) * c.one
