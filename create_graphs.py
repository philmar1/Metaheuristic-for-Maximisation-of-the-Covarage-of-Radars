import numpy as np
import matplotlib.pyplot as plt

import csv

import os, sys
from os import listdir
from os.path import isfile, join

########################################################################
# get all files related to one method
########################################################################

def get_files(algo_name, mypath = "/Users/macphilou/Documents/3A/IA/IA308 Meta heuristique/sho_18-19_vf/solvers/epsilon"):
    files = [file for file in listdir(mypath) if isfile(join(mypath, file)) and file.startswith(algo_name + '.') and file.endswith(".csv") ]
    for file in files :
        return(os.path.join(mypath, file))


def get_max_from_runs(file_path) :
    """get a list of each max for all runs"""
    best_iter, max_cover = 0, 0
    best_iters, max_covers = [],[]
    counter_row = 0
    new_run = True
    with open(file_path) as csvfile :
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == '1' and counter_row >= 1 :
                best_iters.append(best_iter)
                max_covers.append(max_cover)
                best_iter, max_cover = 0, 0
            elif not(row[0].startswith('# ')) and float(row[1]) > max_cover :
                max_cover = float(row[1])
                best_iter = int(row[0])
            counter_row += 1
    csvfile.close()
    result = list(zip(best_iters, max_covers))
    result.sort(key = lambda t: t[0])
    result.pop(0)
    return result

def cumulate_score(result,upper_lim, ratio = 0.90) :
    """return the cumulated score"""
    iter = len(result)
    cumulated_score = [0] * 300
    for r in result :
        if r[1] >= ratio * upper_lim :
            for x in range(r[0],300) :
                cumulated_score[x] += 1
    for x in range(300) :
        cumulated_score[x] /= iter
    return cumulated_score

def plot(cumulated_scores,labels) :
    "plot the cumulated score"""
    for cumulated_score in cumulated_scores :
        plt.plot(cumulated_score)
    plt.title("ERT for {} runs. Max_cover = 780, delta = 10%".format(labels['Nb_runs']))
    plt.legend([x for x in labels['Algo']])
    plt.xlabel("Nb of calls of the objective function")
    plt.show()

########################################################################
# Run
########################################################################

if __name__ == "__main__" :
    solvers = ["num_greedy","bit_greedy","bit_annealing","bit_random","bit_annealing_improved"]
    
    # get and process file
    cumulated_scores, algos = [], []
    for solver in solvers :
        file_path = get_files(solver)
        if file_path != None :
            print(solver,file_path)
            algos.append(solver)
            cumulated_score = cumulate_score(get_max_from_runs(file_path),780)
            cumulated_scores.append(cumulated_score)
    plot(cumulated_scores,{'Algo' : algos, 'Nb_runs' : len(cumulated_score)})
