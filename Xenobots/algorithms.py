import abc
import random

import numpy as np


class Algorithm(abc.ABC):

    def __init__(self, n_dim, fitness_func):
        self.n_dim = n_dim
        self.fitness_func = fitness_func
        self.history = []

    @abc.abstractmethod
    def get_best(self):
        pass

    @abc.abstractmethod
    def learn(self, n_iter):
        pass


class GeneticAlgorithm(Algorithm):

    def __init__(self, n_dim, fitness_func, n_pop=25):
        super().__init__(n_dim, fitness_func)
        self.pop = np.random.random(size=(n_pop, n_dim))
        self.pop[:, 0] = self.pop[:, 0] * 480 + 20
        self.pop[:, 1] = self.pop[:, 1] * 19500 + 500
        self.fitness_list = self.eval_fitness(individuals=self.pop)
        self.history = []

    def _tournament_select(self, k, n=5):
        selected = []
        for _ in range(k):
            contenders = np.random.choice(np.arange(len(self.pop)), size=n, replace=False)
            selected.append(self.pop[np.argmax(self.fitness_list[contenders])])
        return selected

    def _cx(self):
        parents = self._tournament_select(k=2)
        child = parents[0].copy()
        for k, (i, j) in enumerate(zip(parents[0], parents[1])):
            child[k] = i if random.random() > 0.5 else j
        return child  # self._mutation(parent=child)

    def _mutation(self, parent=None):
        child = self._tournament_select(k=1)[0].copy() if parent is None else parent
        idx = random.randint(0, len(child) - 1)
        if idx == 0:
            val = np.random.normal(0.0, scale=40)
        elif idx == 1:
            val = np.random.normal(0.0, scale=1000)
        else:
            val = np.random.normal(0.0, scale=0.1)
        child[idx] += val
        return child

    def _reproduce(self):
        children = []
        while len(children) < len(self.pop):
            if random.random() > 0.8:
                child = self._mutation()
            else:
                child = self._cx()
            children.append(child)
        return np.array(children)

    def _survival_select(self, offspring, fitness_list):
        if not len(offspring) or not len(fitness_list):
            return
        self.fitness_list = np.concatenate([self.fitness_list, fitness_list], axis=0)
        survival_idx = np.argsort(self.fitness_list)[-len(self.pop):]
        self.pop = np.concatenate([self.pop, offspring], axis=0)[survival_idx]
        self.fitness_list = self.fitness_list[survival_idx]

    def eval_fitness(self, individuals):
        return np.array([self.fitness_func(ind) for ind in individuals])

    def get_best(self):
        return self.pop[np.argmax(self.fitness_list)]

    def learn(self, n_iter):
        for i in range(n_iter):
            offspring = self._reproduce()
            fitness = self.eval_fitness(individuals=offspring)
            self._survival_select(offspring=offspring, fitness_list=fitness)
            self.history.append(np.max(self.fitness_list))
            print(i)
        return self.get_best()
