import numpy as np
import pandas as pd
import random
def split_X_y(file_name):
    allData = pd.read_csv(file_name, header=None)
    X = allData.iloc[:, :-1].values
    label = allData.iloc[:, -1:].values
    return X, label

def augment(X):
    t = np.ones((len(X), len(X[0]) + 1))
    t[:, :-1] = X

    return t

def initialise(num_genes, num_weights):
    return np.random.uniform(low=-4.0, high=4.0, size=(num_genes, num_weights))


def fitness(X, label, programs):
    scores = []
    label = label.reshape(1, -1)[0]
    for g in programs:
        #sum and multiply the program with each set of data
        sigma = np.matmul(X, g)
        a = (sigma >= 0)
        b = label.reshape((1, -1))
        #the score of each programs
        scores.append(np.sum(a == b) / len(X))

    return scores

def copy(rank_index, genes, num_top10):
    #copy top 10% high quality genes

    top10 = rank_index[:int(num_top10)]
    return genes[top10]


def crossover(high_quality_programs, num_rest_programs):
    children = []

    while (len(children) < num_rest_programs):
        # Create two different random numbers
        randomNumber = random.sample(range(len(high_quality_programs[0])), 2)

        # Pick two genes as mother and father randomly with above indices
        mother = high_quality_programs[randomNumber[0]].copy()
        father = high_quality_programs[randomNumber[1]].copy()

        # Compare each element
        compared_element = np.array(mother) == np.array(father)

        if np.all(compared_element):
            pass
        else:
            child = []

            # randomly pick gene from mother or father for crossover
            for i in range(len(mother)):
                if np.random.randint(2): #random generate 0 or 1, for 1: pick mother, for 0: pick father
                    child.append(mother[i])
                else:
                    child.append(father[i])

            children.append(child)

    return children

def mutation(programs):
    #get ramdom programs
    #random row
    row = random.randint(0, len(programs) -1)
    programs[row] *= np.power(-1, row) + 0.5
    return programs

num_programs = 1000
np.random.seed(663)
X, label = split_X_y('gp-training-set.csv')
X = augment(X)
# initialise
programs = initialise(num_programs, len(X[0]))
for i in range(500):
    # fitnessS
    f = fitness(X, label, programs)
    rank_index = np.argsort(f)[::-1][:]
    print('accuracy, [Weights, -Theta]')
    print(f[rank_index[0]])
    print(programs[rank_index[0]])
    print('')
    if f[rank_index[0]] > 0.97:
        break;
    copy_programs = copy(rank_index, programs, num_programs * 0.10)
    #copy Top 10% program to generation (n+1)
    programs[:len(copy_programs)] = copy_programs
    programs[len(copy_programs):] = crossover(copy_programs, num_programs - len(copy_programs))

    programs = mutation(programs)

print('Final accracy, [Weights, -Theta]')
print(f[rank_index[0]])
print(programs[rank_index[0]])