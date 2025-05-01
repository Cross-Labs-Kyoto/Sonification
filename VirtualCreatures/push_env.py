#!/usr/bin/env python3
from environments import PushEnv, Action
from loguru import logger


OUT_DISP = True
DT = 1 / 30
SIZE = 800


# Instantiate a push environment
env = PushEnv(SIZE, DT, out=OUT_DISP)

# Initialize the environment
obs = env.reset()
logger.debug(f'Init obs: {obs}')

# Run the environment for 200 steps
try:
    for _ in range(200):
        act = Action()
        obs, is_final = env.step(act)
        logger.debug(f'Obs: {obs}, Final: {is_final}')

        if is_final:
            break
except KeyboardInterrupt:
    pass
finally:
    # Close the environment
    env.close()
