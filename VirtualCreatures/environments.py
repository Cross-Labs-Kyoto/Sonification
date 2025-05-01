#!/usr/bin/env python3
from itertools import chain
from collections import namedtuple
from multiprocessing import Process, Queue, Event
from queue import Empty
from time import sleep

import numpy as np
import pymunk as pk
import pygame as pg

from loguru import logger


def display(q, end_evt, size, fps=30):
    """Displays the environment's state in a separate window.

    Parameters
    ----------
    q: Queue
        A queue from which to retrieve the current environment's state

    end_evt: Event
        An event to signal that the process should or has ended (when exception raised)

    size: int
        The environment's size in pixels. The environment is assumed to be square

    fps: int
        The rate at which to update the display

    """

    # Initialize display
    pg.init()
    screen = pg.display.set_mode((size, size))
    clock = pg.time.Clock()

    try:
        while not end_evt.is_set():
            # Get the environment's state
            try:
                state = q.get(block=False)
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
            clock.tick(fps)

            # Handle events
            for evt in pg.event.get():
                if evt.type == pg.QUIT or (evt.type == pg.KEYDOWN and evt.key in [pg.K_q, pg.K_ESCAPE]):
                    end_evt.set()

    except KeyboardInterrupt:
        pass
    finally:
        if not end_evt.is_set():
            # Let everyone know something went wrong
            end_evt.set()

        # Gracefully close pygame and all its module
        pg.quit()

        # Empty the queue
        while not q.empty():
            try:
                q.get(block=False)
            except Empty:
                break


Action = namedtuple('Action', ['acc_agt_1', 'acc_agt_2'], defaults=[(0, 0), (0, 0)])
Observation = namedtuple('Observation', ['pos_agt_1', 'vel_agt_1', 'pos_agt_2', 'vel_agt_2', 'pos_push', 'vel_push', 'pos_goal'], defaults=[])


class PushEnv(object):
    """Defines a simple cooperative push task and physics based environment."""

    def __init__(self, size, dt, goal_pos=None, push_pos=None, out=False):
        """Keeps track of all required parameters defining the current environment.

        Parameters
        ----------

        """

        # Manage the display if necessary
        self._disp = out
        if out:
            self._disp_q = Queue()
            self._disp_end_evt = Event()
            self._disp_proc = Process(target=display, args=(self._disp_q, self._disp_end_evt, size), kwargs={'fps': int(1 / dt)})
            self._disp_proc.start()


        # Store the environment's parameters
        self.size = size
        self.goal_pos = goal_pos
        self.push_pos = push_pos
        self._dt = dt
        self._initialized = False

    def reset(self):
        """Rebuilds the environment, effectively resetting it for the next experiment.

        Returns
        -------
        Observation
            A namedtuple corresponding to the observations for the current environment.

        """

        # Define a space in which the simulation will happen
        self._env = pk.Space()
        self._env.gravity = (0, 0)  # We stay in the X, Y plane

        # Define two groups to filter out the goal from the rest of the elements
        grp_0 = pk.ShapeFilter(group=0)
        grp_1 = pk.ShapeFilter(group=1)

        # Create walls to delineate environment
        static_body = self._env.static_body
        self._walls = [pk.Segment(static_body, a=(0, 0), b=(self.size, 0), radius=1),
                 pk.Segment(static_body, a=(self.size, 0), b=(self.size, self.size), radius=1),
                 pk.Segment(static_body, a=(self.size, self.size), b=(0, self.size), radius=1),
                 pk.Segment(static_body, a=(0, self.size), b=(0, 0), radius=1)]

        # And make them frictionless and non-bouncy
        for w in self._walls:
            w.elasticity = 0
            w.friction = 0
            w.filter = grp_0

        # Create two agents
        self._agts = []
        for idx in range(2):
            shape = pk.Circle(pk.Body(), radius=self.size * 0.05)
            shape.friction = 0
            shape.elasticity = 0
            shape.mass = 1
            shape.filter = grp_0
            self._agts.append(shape)


        # Create a pushable ball or cube
        self._pushable = pk.Circle(pk.Body(), radius=self.size * 0.1)
        self._pushable.elasticity = 0
        self._pushable.friction = 0
        self._pushable.mass = 1
        self._pushable.filter = grp_0

        # Create the goal in a random position
        self._goal = pk.Circle(pk.Body(body_type=pk.Body.STATIC), radius=self.size * 0.15)
        self._goal.filter = grp_1
        if self.goal_pos is not None:
            self._goal.body.position = self.goal_pos
        else:
            self._goal.body.position = (np.random.random((2, )) * self.size).tolist()

        # Randomly set the position and velocity of agents and pushable
        for agt in self._agts:
            agt.body.position = (np.random.random((2,)) * self.size).tolist()
            agt.body.velocity = (np.random.random((2, )) * 100).tolist()  # pixels / sec

        if self.push_pos is None:
            self._pushable.body.position = (np.random.random((2,)) * self.size).tolist()
        else:
            self._pushable.body.position = self.push_pos
        self._pushable.body.velocity = (np.random.random((2,)) * 100).tolist()

        # Add everything to the simulation space
        for el in chain(self._agts, self._walls, [self._pushable]):
            if el.body.body_type != pk.Body.STATIC:
                self._env.add(el, el.body)
            else:
                self._env.add(el)

        # Set initialized flag
        self._initialized = True

        # Display the environment if necessary
        if self._disp:
            self._display_env()

        # Return observations corresponding to initial state
        return self.observe()

    def step(self, action):
        """Executes the given `action` and moves the environment forward in time.

        Parameters
        ----------
        action: Action
            A namedtuple containing the acceleration for each agent in the environment.

        Returns
        -------
        Observation, bool
            A namedtuple corresponding to the observations for the current environment, and a boolean flag indicating whether the environment reached a final state.

        """

        if not self._initialized:
            logger.error('Environment has not been reset. No action can be taken.')
        else:
            # Apply the given accelerations as forces to each agent's center of gravity
            # Since F = ma and m = 1, then F = a
            # By default force is applied to center of gravity
            for idx, agt in enumerate(self._agts):
                self._agts[idx].body.apply_force_at_local_point(force=getattr(action, f'acc_agt_{idx + 1}'))

            # Move the environment forward by a step
            self._env.step(self._dt)

            # If necessary send data to the display
            if self._disp:
                self._display_env()

        # Return a tuple of observation for the new environment's state, and a flag indicating the end of the task
        return self.observe(), self.is_final()

    def _display_env(self):
        """Sends environment state to display process."""

        # Only send data to the display if it is alive
        if not self._disp_end_evt.is_set():
            # Compile the environment's state into the format expected by the display process
            data = {'agents': [(agt.body.position, agt.body.velocity, agt.radius) for agt in self._agts],
                    'pushable': (self._pushable.body.position, self._pushable.body.velocity, self._pushable.radius),
                    'goal': (self._goal.body.position, self._goal.radius)}
            # Weee-!
            self._disp_q.put(data)

    def is_final(self, thres=0.5):
        """Checks if environment has reached a final state (i.e.: pushable and goal intersect more than `thres` percent).

        Parameters
        ----------
        thres: float
            The amount of overlap between the pushable and goal above which the task is considered as completed.

        Returns
        -------
        bool
            True if the task has reached a final state. False otherwise.

        """

        # If the environment has not been initialized the goal cannot be reached
        if not self._initialized:
            return False

        # Make sure the threshold is in the range [0, 1]
        thres = max(0, min(1, thres))

        # Return true if distance between centers is bellow a certain fraction of both radii
        if np.linalg.norm(self._goal.body.position - self._pushable.body.position).item() / (self._goal.radius + self._pushable.radius) <= thres:
            # Reset the init flag, so environment does not get modified after final state
            self._initialized = False
            return True
        else:
            return False

    def observe(self):
        """Observes the current environment's state.

        Returns
        -------
        Observation
            A namedtuple corresponding to the observations for the current environment.
            None if the environment is not initialized.

        """

        # If the environment is not initialized, there is nothing to observe
        if not self._initialized:
            return None

        # Observe the environment
        obs = {'pos_push': self._pushable.body.position, 'vel_push': self._pushable.body.velocity, 'pos_goal': self._goal.body.position}
        for idx, agt in enumerate(self._agts):
            obs[f'pos_agt_{idx + 1}'] = agt.body.position
            obs[f'vel_agt_{idx + 1}'] = agt.body.velocity

        # Return an observation object
        return Observation(**obs)

    def close(self):
        """Terminates the display process if necessary."""

        if self._disp:
            logger.debug('Waiting for the display to consume the queue.')
            while not self._disp_q.empty():
                sleep(self._dt)

            logger.debug('Asking the display process to end.')
            self._disp_end_evt.set()

            logger.debug('Waiting for the display process to exit.')
            self._disp_proc.join()
