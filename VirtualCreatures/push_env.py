#!/usr/bin/env python3
from multiprocessing import Process, Event, Queue
from queue import Empty
from itertools import chain
import numpy as np
import pymunk as pk
import pygame as pg


DEBUG = True
FPS = 30
SIZE = 800

END_EVT = Event()
DISP_Q = Queue()


def display():
        # Initialize display
        pg.init()
        screen = pg.display.set_mode((SIZE, SIZE))
        clock = pg.time.Clock()

        try:
            while not END_EVT.is_set():
                # Get the environment's state
                try:
                    state = DISP_Q.get(block=False)
                except Empty:
                    continue

                # Reset the screen
                screen.fill((255, 255, 255))

                # Draw the agents
                for a in state['agents']:
                    pg.draw.circle(screen, (255, 0, 0), a[0], radius=a[2])
                    pg.draw.line(screen, (0, 0, 0), a[0], a[0] + a[1], width=2)

                # Draw the pushable
                p = state['pushable']
                pg.draw.circle(screen, (0, 0, 255), p[0], radius=p[2])
                pg.draw.line(screen, (0, 0, 0), p[0], p[0] + p[1], width=2)

                # Draw the goal
                g = state['goal']
                pg.draw.circle(screen, (0, 255, 0), g[0], radius=g[1], width=2)

                # Actually display elements
                pg.display.flip()
                clock.tick(FPS)

                # Handle events
                for evt in pg.event.get():
                    if evt.type == pg.QUIT or (evt.type == pg.KEYDOWN and evt.key in [pg.K_q, pg.K_ESCAPE]):
                        END_EVT.set()

        except KeyboardInterrupt:
            pass
        finally:
            if not END_EVT.is_set():
                END_EVT.set()

            # Gracefully close pygame and all its module
            pg.quit()


def create_env():
    # Define a space in which the simulation will happen
    space = pk.Space()
    space.gravity = (0, 0)  # We stay in the X, Y plane

    # Define two groups to filter out the goal from the rest of the elements
    grp_0 = pk.ShapeFilter(group=0)
    grp_1 = pk.ShapeFilter(group=1)

    # Create walls to delineate environment
    static_body = space.static_body
    walls = [pk.Segment(static_body, a=(0, 0), b=(SIZE, 0), radius=1),
             pk.Segment(static_body, a=(SIZE, 0), b=(SIZE, SIZE), radius=1),
             pk.Segment(static_body, a=(SIZE, SIZE), b=(0, SIZE), radius=1),
             pk.Segment(static_body, a=(0, SIZE), b=(0, 0), radius=1)]

    # And make them frictionless and non-bouncy
    for w in walls:
        w.elasticity = 0
        w.friction = 0
        w.filter = grp_0

    # Create two agents
    agts = []
    for idx in range(2):
        shape = pk.Circle(pk.Body(), radius=SIZE * 0.05)
        shape.friction = 0
        shape.elasticity = 0
        shape.mass = 1
        shape.filter = grp_0
        agts.append(shape)

    # Create a pushable ball or cube
    pushable = pk.Circle(pk.Body(), radius=SIZE * 0.1)
    pushable.elasticity = 0
    pushable.friction = 0
    pushable.mass = 1
    pushable.filter = grp_0

    # Create the goal in a random position
    goal = pk.Circle(pk.Body(body_type=pk.Body.STATIC), radius=SIZE * 0.15)
    goal.body.position = (np.random.random((2, )) * SIZE).tolist()
    goal.filter = grp_1

    # Randomly set the position and velocity of agents and pushable
    for agt in agts:
        agt.body.position = (np.random.random((2,)) * SIZE).tolist()
        agt.body.velocity = (np.random.random((2, )) * 100).tolist()  # pixels / sec

    pushable.body.position = (np.random.random((2,)) * SIZE).tolist()
    pushable.body.velocity = (np.random.random((2,)) * 100).tolist()

    # Add everything to the simulation space
    for el in chain(agts, walls, [pushable]):
        if el.body.body_type != pk.Body.STATIC:
            space.add(el, el.body)
        else:
            space.add(el)

    # Return a reference to the environment and every shapes in it
    return space, agts, pushable, goal


def is_final(goal, pushable, thres=0.5):
    # Return true if distance between centers is bellow a certain fraction of the both radii
    max_dist = goal.radius + pushable.radius
    dist = np.linalg.norm(goal.body.position - pushable.body.position).item()

    if dist / max_dist <= thres:
        return True
    else:
        return False


if __name__ == "__main__":
    # Define the time between simulation loops
    dt = 1 / FPS
    # Create the simulation environment
    env, agts, pushable, goal = create_env()

    if DEBUG:
        # Start the display
        DISP_THR = Process(target=display)
        DISP_THR.start()

    # Run the simulation
    try:
        while not END_EVT.wait(dt):
                # Move a step forward in time
                env.step(dt)

                if DEBUG:
                    # Send new state to display thread
                    # It should be noted that velocities are in the body's frame of reference
                    data = {'agents': [(agt.body.position, agt.body.velocity, agt.radius) for agt in agts],
                            'pushable': (pushable.body.position, pushable.body.velocity, pushable.radius),
                            'goal': (goal.body.position, goal.radius)}
                    DISP_Q.put(data)

    except KeyboardInterrupt:
        pass
    finally:
        if DEBUG:
            if not END_EVT.is_set():
                END_EVT.set()

            while not DISP_Q.empty():
                try:
                    DISP_Q.get(block=False)
                except Empty:
                    break
            if hasattr(DISP_Q, 'close'):
                DISP_Q.close()

            DISP_THR.join()
