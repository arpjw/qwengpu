"""Fixed lane commit commands and model-derived memory write inputs."""

COMMIT_BASE = 2688
COMMIT_COUNT = 8
ATOMIC_HIGH_COMMIT_BASE = 2696
INTERNAL_STORES_LAYOUT = "qwengpu-resident-model-stores-9b-v1"


def key_matches(c, bits, base, enabled):
    from chipformer.dsl.graph import Expr

    return sum(
        (value if base >> i & 1 else c.one - value for i, value in enumerate(bits)),
        Expr(),
    ) + enabled - len(bits) * c.one
