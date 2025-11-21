import os

STATIC_MODE = os.environ.get("STATIC_MODE", "false").lower()
if STATIC_MODE in ["1", "true", "on"]:
    STATIC_MODE = True
else:
    STATIC_MODE = False

MAX_NUM_CONFIG = os.environ.get("MAX_NUM_CONFIG", 2)
MAX_NUM_STAGES = os.environ.get("MAX_NUM_STAGES", 4)


print(f"ninetoothed use static mode: {STATIC_MODE}")
print(f"ninetoothed max num config: {MAX_NUM_CONFIG}")
print(f"ninetoothed max num stages: {MAX_NUM_STAGES}")
