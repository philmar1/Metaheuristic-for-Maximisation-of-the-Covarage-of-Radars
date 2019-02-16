import numpy as np
import matplotlib.pyplot as plt
from math import floor

from codes import *

########################################################################
# Interface
########################################################################

if __name__=="__main__":
    import argparse

    # Dimension of the search space.
    d = 2

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=4, type=int,
            help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
            help="Sensors' range (as a fraction of domain width)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
            help="Domain width (a number of cells)")

    can.add_argument("-i", "--iters", metavar="NB", default=300, type=int,
                     help="Maximum number of iterations") #initial : 500

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_greedy","bit_greedy","bit_annealing","bit_random","bit_annealing_improved"]
    can.add_argument("-m", "--solver", metavar="NAME", choices=solvers, default="bit_greedy",
            help="Solver to use, among: "+", ".join(solvers))

    can.add_argument("-t", "--target", metavar="VAL", default=30*30, type=float,
            help="Objective function value target")

    can.add_argument("-y", "--steady-delta", metavar="NB", default=50, type=float,
                     help="Stop if no improvement after NB iterations")  #initial : 50
            
    can.add_argument("-e", "--steady-epsilon", metavar="DVAL", default=0, type=float,
            help="Stop if the improvement of the objective function value is lesser than DVAL")
    
    # ---- START HERE
    # my arguments :
    
    can.add_argument("-temp", "--temperature", metavar="NB", default=100, type=float,
            help="Temperature for the annealing algo")
            
    can.add_argument("-rrs", "--ratio_random_search", metavar="NB", default=0.4, type=float,
            help="Ratio of the random search")
    
    can.add_argument("-runs", "--nb_runs", metavar="NB", default=1, type=int,
                     help="Number of runs")

    the = can.parse_args()


    # Minimum checks
    assert(0 < the.nb_sensors)
    assert(0 < the.sensor_range <= 1)
    assert(0 < the.domain_width)
    assert(0 < the.iters)

    # Seed option in case you would start some runs in parallel.
    np.random.seed(the.seed)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth = np.inf)


    # Common termination and checkpointing.
    history = []
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = the.iters),
                    make.iter(iters.save,
                        filename = the.solver+".csv",
                        fmt = "{it} ; {val} ; {sol}\n"),
                    make.iter(iters.log,
                        fmt="\r{it} {val}"),
                    make.iter(iters.history,
                        history = history),
                    make.iter(iters.target,
                        target = the.target),
                    iters.steady(the.steady_delta, the.steady_epsilon)
                ]
            )
            
    # Erase the previous file.
    with open(the.solver+".csv", 'a') as fd:
        fd.write("# {} {}\n".format(the.solver,the.domain_width))

    val,sol,sensors = None,None,None
    if the.solver == "num_greedy":
        print(" defaut mode : {}".format(the.solver))
        for i in range(the.nb_runs) :
            print("run {} out of {}".format(i,the.nb_runs),floor(100*i/the.nb_runs)," % done", end = '\r')
            val,sol = algo.greedy(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.domain_width/10, domain_width = the.domain_width),
                iters
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_greedy":
        print(" mode : {}".format(the.solver))
        for i in range(the.nb_runs) :
            print("run {} out of {}".format(i,the.nb_runs),floor(100*i/the.nb_runs)," % done", end = '\r')
            val,sol = algo.greedy(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.domain_width/10, domain_width = the.domain_width),
                iters
            )
        sensors = bit.to_sensors(sol)


    elif the.solver == "bit_random":
        print(" mode : {}".format(the.solver))
        for i in range(the.nb_runs) :
            print("run {} out of {}".format(i,the.nb_runs),floor(100*i/the.nb_runs)," % done", end = '\n')
            val,sol = algo.random_solver(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                iters
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "bit_annealing":
        print(" mode : {}".format(the.solver))
        for i in range(the.nb_runs) :
            print("run {} out of {}".format(i,the.nb_runs),floor(100*i/the.nb_runs)," % done", end = '\r')
            val,sol = algo.annealing(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.domain_width/10, domain_width = the.domain_width),
                    iters,
                    temp = the.temperature,
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "bit_annealing_improved":
        print(" mode : {}".format(the.solver))
        for i in range(the.nb_runs) :
            print("run {} out of {}".format(i,the.nb_runs),floor(100*i/the.nb_runs)," % done", end = '\n')
            val,sol = algo.annealing_improved(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.domain_width/10, domain_width = the.domain_width),
                iters,
                nb_iter = the.iters,
                ratio_random_search = the.ratio_random_search,
                temp = the.temperature,
            )
            sensors = bit.to_sensors(sol)

# ---- END HERE

    # Fancy output.
    print("\n{} : {}".format(val,sensors))

    shape=(the.domain_width, the.domain_width)

    fig = plt.figure()

    if the.nb_sensors ==1 and the.domain_width <= 50:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        f = make.func(num.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range * the.domain_width)
        plot.surface(ax1, shape, f)
        plot.path(ax1, shape, history)
    else:
        ax2=fig.add_subplot(111)

    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)

    plt.show()
