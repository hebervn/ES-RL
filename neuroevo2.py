"""
neuroevo.py
"""
__author__ = "giorgio@ac.upc.edu"
__credits__ = "https://github.com/yanpanlau"


from Environment import OmnetBalancerEnv
from Environment import OmnetLinkweightEnv
import numpy as np
import tensorflow as tf
import json
import os
import multiprocessing
import threading
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import array
import random
from time import time
import multiprocessing
import uuid
import pickle

from ActorNetwork import ActorNetwork
from helper import setup_run, setup_exp

#tf.get_logger().setLevel('ERROR')
#tf.autograph.set_verbosity(1)

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with open('DDPG.json') as jconfig:
    DDPG_config = json.load(jconfig) # LÊ O ARQUIVO DE CONFIGURAÇÃO
DDPG_config['EXPERIMENT'] = setup_exp() # CRIA UMA PASTA PARA SALVAR OS DADOS DA EXECUÇÃO

EPISODE_COUNT = DDPG_config['EPISODE_COUNT']
MAX_STEPS = DDPG_config['MAX_STEPS']


def create_environment(key=''):
    folder = setup_run(DDPG_config, key)
    # Generate an environment
    if DDPG_config['ENV'] == 'balancing':
        env = OmnetBalancerEnv(DDPG_config, folder)
    elif DDPG_config['ENV'] == 'label':
        env = OmnetLinkweightEnv(DDPG_config, folder)

    return env


def update_model_weights(model, new_weights):
    """
    Updates the network with new weights after they have been stored in one
    flat parameter vector
    """
    accum = 0
    for layer in model.layers:
        current_layer_weights_list = layer.get_weights()
        new_layer_weights_list = []
        for layer_weights in current_layer_weights_list:
            layer_total = np.prod(layer_weights.shape)
            new_layer_weights_list.append(
                new_weights[accum:accum + layer_total].
                reshape(layer_weights.shape))
            accum += layer_total
        layer.set_weights(new_layer_weights_list)


def simulate(individual):
    key = str(uuid.uuid1())
    env = create_environment(key)

    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    action_dim, state_dim = env.a_dim, env.s_dim
    actor = ActorNetwork(sess, state_dim, action_dim, DDPG_config)
    update_model_weights(actor.model, np.asarray(individual))

    state = env.reset()
    delays = []

    for i in range(MAX_STEPS):
        action = actor.model.predict(state.reshape(1, state.shape[0]))[0]
        print("STEP - {}".format(i))
        state, reward, _ = env.step(action)

        # environment returns negative reward
        delays.append(reward)

    env.end()
    return delays


def fitness(individual):
    """
    Calcula fitness do individuo
    Soma do delay executando MAX_STEPS ações no simulador
    """

    delays = simulate(individual)
    fitness = sum(delays)
    print(fitness)
    return (fitness,)


""" def selAverage(individuals, k, fit_attr='fitness'):
    ind = individuals[0]
    print("K = {}".format(k))
    for j in range(len(ind)):
        ind[j] *= 1 / abs(getattr(ind, fit_attr).values[0])
    print("Antes {} - Depois {}".format(ind[j], individuals[0][j]))
    sum_fitness = 1 / abs(getattr(ind, fit_attr).values[0])
    print("SUM FIT: {}".format(sum_fitness))
    for i in range(1, len(individuals)):
        fitness = abs(getattr(individuals[i], fit_attr).values[0])
        for j in range(len(individuals[i])):
            ind[j] += (1 / fitness) * individuals[i][j]
        sum_fitness += 1 / fitness

    for j in range(len(ind)):
        ind[j] /= sum_fitness

    print()
    return [ind for _ in range(k)]
 """

def selAverage(individuals, k, fit_attr='fitness'):
    ind = individuals[0]
    print("K = {}".format(k))
    fitness = abs(getattr(ind, fit_attr).values[0])
    for j in range(len(ind)):
        ind[j] = ind[j] * fitness
    sum_fitness = fitness
    
    for i in range(1, len(individuals)):
        fitness = abs(getattr(individuals[i], fit_attr).values[0])
        for j in range(len(individuals[i])):
            ind[j] += fitness * individuals[i][j]
        sum_fitness += fitness

    for j in range(len(ind)):
        ind[j] /= sum_fitness
    print("Mean Fitness: {}".format(sum_fitness/len(individuals)))
    return [ind for _ in range(k)]

def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


#### DEAP Config #####
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array.array, typecode="d")

IND_SIZE = 21420
MIN_VALUE, MAX_VALUE = -1.0, 1.0
MIN_STRATEGY, MAX_STRATEGY = 1e-7, 0.5
MU = 1
LAMBDA = 8

toolbox = base.Toolbox()

# generation functions
#toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
#              IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#toolbox.register("mutate", tools.mutESLogNormal, c=1, indpb=0.1)
#toolbox.register("mate", tools.cxESTwoPoint)
#toolbox.register("evaluate", fitness)
#toolbox.register("select", tools.selTournament, tournsize=16)
######################
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
              IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", tools.mutESLogNormal, c=1, indpb=0.1)
toolbox.register("mate", tools.cxESTwoPoint)
toolbox.register("evaluate", fitness)
toolbox.register("select", selAverage)

def playGame(DDPG_config, train_indicator=1, verbose=False):    #1 means Train, 0 means simply Run
    if DDPG_config['RSEED'] == 0:
        DDPG_config['RSEED'] = None
    np.random.seed(DDPG_config['RSEED'])
    random.seed(DDPG_config['RSEED'])

    if train_indicator == 0:
        with open('neuroevo_solutions/solution_1625177454.784881.pkl', 'rb') as f:
            sol = pickle.load(f)

        ind = sol['halloffame'][0]

        with open('neuroevo_results_test/results_1625177454.784881.pkl.txt', 'w') as f:
            for intensity in [125, 250, 375, 500, 625, 750, 875, 1000, 1125, 1250]:
                DDPG_config['TRAFFIC'] = "DIR:traffics/traffic{}/".format(intensity)
                delays = simulate(ind)
                delays = list(map(lambda x: -x, delays))

                print('===== Intensity {} ====='.format(intensity))
                print('Min Delay: {:.3f}'.format(min(delays)))
                print('Max Delay: {:.3f}'.format(max(delays)))
                print('Average Delay: {:.3f}'.format(sum(delays) / len(delays)))

                f.write('===== Intensity {} =====\n'.format(intensity))
                f.write('Min Delay: {:.3f}\n'.format(min(delays)))
                f.write('Max Delay: {:.3f}\n'.format(max(delays)))
                f.write('Average Delay: {:.3f}\n'.format(sum(delays) / len(delays)))

    elif train_indicator == 1:
        pool = multiprocessing.Pool(processes=8)
        toolbox.register("map", pool.map)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)

        pop = toolbox.population(n=MU)

        start_time = time()

        pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                  cxpb=0.0, mutpb=0.9, ngen=EPISODE_COUNT, stats=stats, halloffame=hof, verbose=verbose)

        elapsed_time = time() - start_time

        pool.close()

        with open('neuroevo_solutions/solution_{}.pkl'.format(start_time), 'wb') as f:
            sol = dict(population=pop, logbook=logbook, halloffame=hof)
            pickle.dump(sol, f)

        # best = fitness(hof[0])[0]
        best = hof[0].fitness.values[0]

        with open('neuroevo_results_train/results_{}.txt'.format(start_time), 'w') as f:
            f.write('{}\n'.format(logbook))
            f.write('Best fitness: {:.3f}\n'.format(best))
            f.write('Time: {:.3f} sec.\n'.format(elapsed_time))

        print('Best fitness: {:.3f}'.format(best))
        print('Time: {:.3f} sec.'.format(elapsed_time))


if __name__ == "__main__":
    playGame(DDPG_config, train_indicator=0, verbose=True)
