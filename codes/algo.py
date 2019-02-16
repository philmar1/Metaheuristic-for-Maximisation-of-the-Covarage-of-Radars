
########################################################################
# Algorithms
########################################################################
import numpy as np
from codes import make

def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 0
    while again(i, val, sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol
# DONE add a random solver

def random_solver(func, init, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    i = 1
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def temperature(fraction) :
    """Evolution function of the temperature for both annealing and improved annealing"""
    return max(0.01, min(1, 1 - fraction))

def annealing(func,init,neighb, again, nb_iter = 300, temp = 1) :
    """ If default = true : the neighboor doesn't depend on temp
        else, we change the neighboor with temp"""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    current_val, current_sol = best_val,best_sol
    i = 1
    while again(i, current_val, current_sol):
        sol = neighb(current_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val, best_sol = val, sol
            current_val, current_sol = val, sol
        elif np.random.random() < np.exp(-abs(val-best_val)/temp) :
            current_val, current_sol = val, sol
        i += 1
        fraction = i / float(nb_iter)
        temp = temperature(fraction)
    return best_val, best_sol

# The next version is updated in order to allow a random search at the beginning
def annealing_improved(func,init,neighb, again, nb_iter = 300, ratio_random_search = 0.1, temp = 1) :
    """ If default = true : the neighboor doesn't depend on temp
        else, we change the neighboor with temp"""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val, best_sol
    current_val, current_sol = best_val, best_sol
    i = 1
    while again(i, current_val, current_sol):
        if i <= nb_iter * ratio_random_search :
            sol = init()
            val = func(sol)
            if val >= best_val:
                best_val, best_sol = val, sol
                current_val, current_sol = val, sol
        else :
            sol = neighb(current_sol)
            val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
            if val >= best_val:
                best_val, best_sol = val, sol
                current_val, current_sol = val, sol
            elif np.random.random() < np.exp(-abs(val-best_val)/temp) :
                current_val, current_sol = val, sol
            fraction = (i - nb_iter * ratio_random_search )/ float(nb_iter)
            temp = temperature(fraction)
        i += 1
    return best_val, best_sol


