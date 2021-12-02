import noise
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from core.safari.agent import Agent, Status


def create_world(size):
    shape = (size, size)
    scale = 100.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0

    DEM = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            DEM[i][j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=size,
                repeaty=size,
                base=17,
            )

    DEM -= DEM.min()
    DEM *= 60
    DEM = np.flip(DEM)
    return DEM


if __name__ == "__main__":
    # ! Create the map
    LOCAL_SIZE = 128
    SIZE = LOCAL_SIZE * 4
    DEM = create_world(SIZE)

    # ! Create the Agent
    transform = lambda pos, dem: dem

    AGENT = Agent(
        start=(5, 5, DEM[5, 5]),
        goal=(500, 500, DEM[500, 500]),
        local_size=LOCAL_SIZE,
        world=DEM,
        transform=transform,
        recalculate_threshold=0.8,
    )

    while True:
        status = AGENT.forward()
        path = AGENT.get_path()

        if status == Status.REACHED:
            logger.success("Path reached")

        # ! Display results
        plt.imshow(DEM, cmap="terrain", extent=[0, SIZE, 0, SIZE], origin="lower")
        plt.plot(path[:, 0], path[:, 1], linewidth=2, c="r")
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

        if status == Status.REACHED:
            break