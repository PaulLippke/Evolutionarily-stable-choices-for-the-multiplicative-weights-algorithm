"""
MIT License

Copyright (c) [2025] [Paul Laurids Lippke]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# This code was created for my master thesis. The topic is evolutionarily stable choices for the multiplicative weights algorithm.
# For the results and the overall setting please consider my master thesis.

import random
import copy
import matplotlib.pyplot as plt

num_strategies = 2#global parameter for the number of strategies

def weight_mass(player, weights):
    """
    Computes the sum of all the weights of a player.
    Used later for normalizing the weights of a player.

    player: the player id of the player it sums the weights of
    weights: the weights for both player

    return: sum of weights for a player
    """
    return sum([weights[player][i] for i in range(num_strategies)])

def compute_costs(weights, costs_table):
    """
    For given weights, turns the weights into probabilities and computes the cost for each strategy according to a cost table.

    weights: the weights for the strategies for both players
    costs_table: cost table of the underlying symmetric game (uses the by symmetry simplified form)

    return: costs: list containing the cost for every strategy
    """
    weight_mass1 = weight_mass(0, weights)
    weight_mass2 = weight_mass(1, weights)
    costs = [
        [0 for _ in range(num_strategies)],
        [0 for _ in range(num_strategies)]
    ]
    for i in range(num_strategies):
        costs[0][i] = sum([costs_table[i][j]*weights[1][j] for j in range(num_strategies)])/weight_mass2 # i Zeile j Spalte von Cost Table
        costs[1][i] = sum([costs_table[i][j]*weights[0][j] for j in range(num_strategies)])/weight_mass1
    return costs

def update_weights(weights, costs, eta1, eta2):
    """
    Corresponds to the update step of the multiplicative weights algorithm.
    Updates the weights of two players based on provided costs.
    The first player is updated with learning rate eta1, the second one with learning rate eta2.
    The weights are then normalized to probabilities.

    weights: the weights before the update
    costs: the cost of each strategy for both players
    eta1: learning rate of the first player
    eta2: learning rate of the second player

    return: None (changes weights)
    """
    for i in range(num_strategies):
        weights[0][i] = weights[0][i]*(1-eta1 * costs[0][i])
        weights[1][i] = weights[1][i]*(1-eta2 * costs[1][i])
    weight_mass1 = weight_mass(0, weights)
    weight_mass2 = weight_mass(1, weights)
    for i in range(num_strategies):
        weights[0][i] = weights[0][i]/weight_mass1
        weights[1][i] = weights[1][i]/weight_mass2

def simulate(steps, eta1, eta2, costs_table, init_weights=None):
    """
    This function simulates two games, one with both players using learning rate eta1 playing against each other. The other with the first player using eta1 and the second one eta2.
    The players play a game over multiple rounds with their strategy mix controlled by the multiplicative weights algorithm.
    This method searches for counterexamples of eta1 being an evolutionarily stable strategy (ESC).

    steps: controls the number of rounds the player play
    eta1: learning rate used to update the weights of the two players in the first game, as well as the second player in the second game
    eta2: learning rate used to update the weights of the first player in the second game
    costs_table: cost table of the underlying symmetric game (uses the by symmetry simplified form)
    init_weights: The inital weights for the strategy mix used by the players, if not given uses the discrete uniform distribution

    return: None (prints found counterexamples/can write them in a file)
    """
    if init_weights==None:
        weights1 = [
            [1 for _ in range(num_strategies)],
            [1 for _ in range(num_strategies)]
        ]
        weights2 = [
            [1 for _ in range(num_strategies)],
            [1 for _ in range(num_strategies)]
        ]
    else:
        weights1 =  [
            copy.deepcopy(init_weights),
            copy.deepcopy(init_weights)
        ]
        weights2 =  [
            copy.deepcopy(init_weights),
            copy.deepcopy(init_weights)
        ]
    for i in range(steps):
        #first game of eta1-player against eta1-player
        costs1 = compute_costs(weights1, costs_table)
        update_weights(weights1, costs1, eta1, eta1)
        costs_complete1 = sum([weights1[0][i]*compute_costs(weights1, costs_table)[0][i] for i in range(num_strategies)])
        #second game of eta2-player against eta1-player
        costs2 = compute_costs(weights2, costs_table)
        update_weights(weights2, costs2, eta2, eta1)
        costs_complete2 = sum([weights2[0][i]*compute_costs(weights2, costs_table)[0][i] for i in range(num_strategies)])
        if costs_complete1 > costs_complete2:#if this is the case we have found a counterexample
            print("Counterexample detected")
            if False:#if set to True, all counterexamples will be written in test.txt
                with open("test.txt", "a") as f:
                    f.write(2*'\n'+30*"-"+'\n')
                    f.write("Cost Table: "+str(costs_table)+'\n'+"Eta_1;Eta_2: "+str(eta1)+";"+str(eta2)+'\n'+"Step: "+str(i)+'\n'+"Weights1: " + str(weights1) +" Costs1: "+ str(costs1) + " Costs Complete1: " + str(costs_complete1)+'\n'+"Weights2: " + str(weights2) +" Costs2: "+ str(costs2) + " Costs Complete2: " + str(costs_complete2))
                    f.write('\n'+30*"-"+2*'\n')

def simulateAndSave(steps, eta1, eta2, costs_table, init_weights=None):
    """
    This function simulates two games, one with two players using learning rate eta1 playing against each other. The other with the first player using eta1 and the second one eta2.
    The players play a game over multiple rounds with their strategy mix controlled by the multiplicative weights algorithm.
    This method then logs the results in a file.

    steps: controls the number of rounds the player play
    eta1: learning rate used to update the weights of the two players in the first game, as well as the second player in the second game
    eta2: learning rate used to update the weights of the first player in the second game
    costs_table: cost table of the underlying symmetric game (uses the by symmetry simplified form)
    init_weights: The inital weights for the strategy mix used by the players, if not given uses the discrete uniform distribution

    return: None (creates a csv file with the weights and costs over all iterations)
    """
    if init_weights==None:
        weights1 = [
            [1 for _ in range(num_strategies)],
            [1 for _ in range(num_strategies)]
        ]
        weights2 = [
            [1 for _ in range(num_strategies)],
            [1 for _ in range(num_strategies)]
        ]
    else:
        weights1 =  [
            copy.deepcopy(init_weights),
            copy.deepcopy(init_weights)
        ]
        weights2 =  [
            copy.deepcopy(init_weights),
            copy.deepcopy(init_weights)
        ]
    f = open("SimulateAndSave.csv", "a")#SimulateAndSave is the file name of the output file
    for i in range(steps):
        costs1 = compute_costs(weights1, costs_table)
        costs2 = compute_costs(weights2, costs_table)
        f.write(str(i)+","+str(weights1[0][0])+","+str(weights1[1][0])+","+str(weights2[0][0])+","+str(weights2[1][0])+"\n")
        update_weights(weights1, costs1, eta1, eta1)
        update_weights(weights2, costs2, eta2, eta1)
    f.close()

def generate_costs_table():
    """
    Generates a random simplified cost table for a finite, 2-player, symmetric game with a special property.
    The cost table has entries from ]-1,1[.
    Also gives two learning rate eta1 and eta2, with eta1 > eta2.

    return: costs_table: cost table for a game with the selected property
            eta1: random learning rate
            eta2: random learning rate smaller than eta1
    """
    cond_random_costs = True#fixed
    cond_dominant_strategy = False#True for a strictly dominant strategy in the first row of the cost table
    cond_better_than_opponent = False#True for better than opponent condition
    cond_better_than_anything = False #True for better than anything condition
    cond_dummy_game = False #True for a dummy game
    cond_coordination_game = False #True for a coordination game
    cond_zero_sum_game = True #True for a zero-sum game
    eta_1 = random.uniform(0.01, 0.49)
    eta_2 = random.uniform(0.01, eta_1)# eta2 < eta1
    costs_table = [ [0 for _ in range(num_strategies)] for _ in range(num_strategies)]
    if cond_random_costs:
        if not cond_zero_sum_game:
            for i in range(0,num_strategies):
                for j in range(0,num_strategies):
                    costs_table[i][j] = random.uniform(-0.99,0.99)#random.uniform(0.01,0.95)#-0.95 for introducing dominant strategy
        else:
            for i in range(0,num_strategies):
                for j in range(0,num_strategies):
                    costs_table[i][j] = random.uniform(-0.99,0.99)#-0.95 for introducing dominant strategy
    if cond_dominant_strategy: #for creation of dominant strategy
        for j in range(num_strategies): 
            necessary_min = min([costs_table[i][j] for i in range(num_strategies)])
            if costs_table[0][j] >= necessary_min:
                costs_table[0][j] = necessary_min-0.01
    if cond_better_than_opponent:#for creation of better than opponent strategy addon
        for j in range(1,num_strategies): 
            costs_table[0][j] = min(costs_table[0][j], costs_table[j][0]-0.01)
    if cond_better_than_anything:#for creation of better than anything strategy addon
        for j in range(num_strategies):
            costs_table[0][j] = min(costs_table[0][j], min([min(costs_table[i]) for i in range(1,num_strategies)]))
    if cond_dummy_game:#for creation of dummy game
        for i in range(num_strategies):
            for j in range(num_strategies):
                costs_table[i][j] = costs_table[0][j]
    if cond_coordination_game:#for creation of coordination game
        for i in range(num_strategies):
            for j in range(i, num_strategies):
                costs_table[i][j] = costs_table[j][i]
    if cond_zero_sum_game:#for creation of zero-sum game
        for i in range(num_strategies):
            costs_table[i][i] = 0
        for i in range(num_strategies):
            for j in range(i, num_strategies):
                costs_table[i][j] = -costs_table[j][i]
    return costs_table, eta_1, eta_2

def generate_costs_table_prisoners_dilemma_2x2():
    """
    Creates a cost table with the same structure like prisoner's dilemma. Additionally creates two learning rates eta1 and eta2 with eta1 > eta2.

    return: costs_table: cost table with prisoner's dilemma type structure
            eta1: random learning rate
            eta2: random learning rate smaller than eta1
    """
    assert num_strategies == 2
    eta_1 = random.uniform(0.01, 0.49)
    eta_2 = random.uniform(0.01, eta_1)
    costs_table = [ [0 for _ in range(num_strategies)] for _ in range(num_strategies)]
    liste = []
    for _ in range(num_strategies*num_strategies):
        liste.append(random.uniform(-0.99, 0.99))
    liste.sort()
    costs_table[0][1] = liste[0]
    costs_table[1][1] = liste[1]
    costs_table[0][0] = liste[2]
    costs_table[1][0] = liste[3]
    return costs_table, eta_1, eta_2


def analyse(steps, eta1, eta2, costs_table, file_name, save_file=1, init_weights=None):
    """
    This method is used to analyse a specific game between two players. The players strategic mix is then controlled by the multiplicative weights algorithm (MWA), where
    the first player uses learning rate eta2 and the second one learning rate eta1.

    steps: number of rounds the two players play
    eta1: learning rate for the first player
    eta2: learning rate for the second player
    costs_table: the simplified cost table of the underlying symmetric game (uses the by symmetry simlified form)
    file_name: name of the csv file with the results
    save_file: boolean if a csv file should be created
    init_weights: initial weights for the MWA if none, the discrete uniform distribution is used.
    
    return: costs1_list: list of costs for all strategies over rounds of the MWA for both players
            weights1_list: list of the weights for all strategies over rounds of the MWA for both players
            cost_complete1_list: list of the cost each player suffers with their strategic mix against the other players, also for all rounds
    """
    weights1_list = []
    costs1_list = []
    costs_complete1_list = []
    if init_weights==None:    
        weights1 = [
            [1 for _ in range(num_strategies)],
            [1 for _ in range(num_strategies)]
        ]
    else:
        weights1 = [
            copy.deepcopy(init_weights),
            copy.deepcopy(init_weights)
        ]
    update_weights(weights1, weights1, 0, 0)
    weights1_list.append(copy.deepcopy(weights1))
    for i in range(steps):
        costs1 = compute_costs(weights1, costs_table)
        costs1_list.append(costs1)
        update_weights(weights1, costs1, eta2, eta1)
        weights1_list.append(copy.deepcopy(weights1))
        costs_complete1 = sum([weights1[0][i]*compute_costs(weights1, costs_table)[0][i] for i in range(num_strategies)])
        costs_complete1_list.append(copy.deepcopy(costs_complete1))
    if save_file:
        with open(file_name + ".csv", "w") as f:
            f.write(str(costs_table)+";"+str(eta1)+";"+str(eta2)+";"+str(costs1_list)+";"+str(weights1_list)+";"+str(costs_complete1_list))#The order of csv variables can be seen here
    return costs1_list, weights1_list, costs_complete1_list

def visualize(cost_list, weights_list, cost_complete_list):
    """
    This method is used to visualize, the result from the analyse method

    cost_list: list of costs for every strategy for both players
    weights_list: list of weights for every strategy and both players
    cost_complete_list: list of costs players suffer with their mix for every given round

    return: None (shows a plot of the first strategy's probability over time)
    """
    cost_l = cost_list
    weights_l = weights_list
    costcom_l = cost_complete_list
    weights_reordered = [[[weights_l[i][player_id][j] for i in range(len(weights_l))]for j in range(num_strategies) ] for player_id in range(0,2)]
    #plt.figure()
    #plt.scatter(weights_reordered[0][0],weights_reordered[0][1])
    #plt.scatter(weights_reordered[1][0],weights_reordered[1][1])      
    #plt.show()
    plt.figure()
    plt.plot(weights_reordered[0][1], label="first player first strategy")
    plt.plot(weights_reordered[1][1], label="second player first strategy")
    plt.show()

def visualize_diff(data1, data2):
    """
    This method generates various plots for two games which are given as data1 and data2. This includes the plots used in the corresponding thesis.

    data1: data from the first game that should be analysed
    data2: data from the second game that should be analysed

    result: None (Various plots: scatter plot for strategies over time, first strategy over time, second strategy over time, each players strategic mix over time, cost of first players over time)
    """
    cost_l1 = data1[0]
    weights_l1= data1[1]
    costcom_l1 = data1[2]
    cost_l2 = data2[0]
    weights_l2 = data2[1]
    costcom_l2 = data2[2]
    weights_reordered1 = [[[ weights_l1[i][player_id][j] for i in range(len(weights_l1))]for j in range(num_strategies) ] for player_id in range(0,2)]
    weights_reordered2 = [[[ weights_l2[i][player_id][j] for i in range(len(weights_l2))]for j in range(num_strategies) ] for player_id in range(0,2)]
    size_of_points = 10
    alpha_of_points = 0.7
    dpi_setting = 500
    plt.figure()
    plt.title("Strategy comparison")#creates a scatter plot of the probabilities over time
    plt.scatter(weights_reordered1[0][0],weights_reordered1[0][1], s=size_of_points, alpha=alpha_of_points, label=r'$p_1 = p_2$')
    #plt.scatter(weights_reordered1[1][0],weights_reordered1[1][1], s=size_of_points, alpha=alpha_of_points, label='1vs1 second')
    plt.scatter(weights_reordered2[0][0],weights_reordered2[0][1], s=size_of_points, alpha=alpha_of_points, label=r'$\tilde{p}_1$')
    plt.scatter(weights_reordered2[1][0],weights_reordered2[1][1], s=size_of_points, alpha=alpha_of_points, label=r'$\tilde{p}_2$')
    plt.xlabel(r'$\text{probability of strategy } \alpha$')
    plt.ylabel(r'$\text{probability of strategy } \beta$')
    plt.legend()
    plt.savefig('currentPicture/alphaVSbeta', dpi= dpi_setting)
    plt.show()
    plt.figure()
    plt.title("Evolution of the strategic mix over time")#creates a plot of the first strategy for different players over time
    plt.plot(weights_reordered1[0][0], label=r'$p_1^j(\alpha) = p_2^j(\alpha)$')
    #plt.plot(weights_reordered1[1][0], label=r'$p_2^\#(\alpha)$')
    plt.plot(weights_reordered2[0][0], label=r'$\tilde{p}_1^j(\alpha)$')
    plt.plot(weights_reordered2[1][0], label=r'$\tilde{p}_2^j(\alpha)$')
    plt.xlabel(r'number of iteration $j$')
    plt.ylabel(r'probability of strategy $\alpha$')
    plt.legend()
    plt.savefig('currentPicture/AlphaOverTime', dpi=dpi_setting)
    plt.show()
    plt.figure()
    plt.title("Evolution of the strategic mix over time")#creates a plot for the second strategy for different players over time
    plt.plot(weights_reordered1[0][1], label=r'$p_1^j(\beta) = p_2^j(\beta)$')
    #plt.plot(weights_reordered1[1][0], label=r'$p_2^\#(\alpha)$')
    plt.plot(weights_reordered2[0][1], label=r'$\tilde{p}_1^j(\beta)$')
    plt.plot(weights_reordered2[1][1], label=r'$\tilde{p}_2^j(\beta)$')
    plt.xlabel(r'number of iteration $j$')
    plt.ylabel(r'$probability of strategy $\beta$')
    plt.legend()
    plt.savefig('currentPicture/BetaOverTime', dpi=dpi_setting)
    plt.show()
    player_prob = ['p_1^j', 'p_2^j', '\\tilde{p}_1^j', '\\tilde{p}_2^j']
    greek_letters = ['\\alpha', '\\beta', '\\gamma']
    title_text = ['Player 1 in $\\eta_1$ vs $\\eta_1$', 'Player 2 in $\\eta_1$ vs $\\eta_1$', 'Player 1 in $\\eta_1$ vs $\\eta_2$', 'Player 2 in $\\eta_1$ vs $\\eta_2$']
    for i in range(4):#creates plots for every players strategies over time
        plt.figure()
        plt.title(r'Development of strategy mix for ' + title_text[i])
        for k in range(num_strategies):
            if i < 2:
                plt.plot(weights_reordered1[i][k], label=r'$'+player_prob[i]+'('+greek_letters[k]+')$')
            else:
                plt.plot(weights_reordered2[i-2][k], label=r'$'+player_prob[i]+'('+greek_letters[k]+')$')
        plt.xlabel(r'number of iteration $j$')
        plt.ylabel(r'probability $'+player_prob[i]+'$')
        plt.legend()
        plt.savefig('currentPicture/StrategyP' +str(i), dpi=dpi_setting)
        plt.show()
    plt.figure()
    plt.title('Development of costs')#plots the costs of the first players in the two games we consider
    plt.plot(costcom_l1, label=r'$C_j(\eta_1 | \eta_1)$')
    plt.plot(costcom_l2, label=r'$C_j(\eta_2 | \eta_1)$')
    plt.xticks([10*l for l in range(int((len(costcom_l1)+1)/10)+1)], [1] + [10*l for l in range(1,int((len(costcom_l1)+1)/10)+1)])
    plt.xlabel(r'number of iteration $j$')
    plt.ylabel('costs at this timepoint')
    plt.legend()
    plt.savefig('currentPicture/TotalCost', dpi=dpi_setting)
    plt.show()

"""
In the following is the code used to generate the plots for the thesis. First are various games that were used.
"""

#COORDINATION GAME COUNTER
# num_strategies = 2
# costs_table, eta_1, eta_2 = [[0.5, -0.5],[-0.5, 0.25]], 0.25, 0.1

#ZERO SUM GAME COUNTER
# num_strategies = 3
# costs_table, eta_1, eta_2 = [[0, 1, -1], [-1, 0, 0.5], [1, -0.5, 0]], 0.5, 0.1

#ZERO SUM 2x2 PROVEIMAGE
# num_strategies = 2
# costs_table, eta_1, eta_2 = [[0, -1],[1, 0]], 0.5, 0.25

#ROCK-PAPER-SCISSOR COUNTER
# num_strategies = 3
# eta_1 = 1/3
# eta_2 = 0.1
# costs_table = [
#     [0, 1, -1],
#     [-1, 0, 1],
#     [1, -1, 0]
# ]

#ROCK-PAPER-SCISSOR COUNTER
# num_strategies = 3
# eta_1 = 0.5
# eta_2 = 0.1
# costs_table = [
#     [0, 1, -1],
#     [-1, 0, 1],
#     [1, -1, 0]
# ] #init_weights = [0.8, 0.1, 0.1]

#DOMINANT STRATEGY COUNTER
# num_strategies = 2
# eta_1 = 1/3
# eta_2 = 0.1
# costs_table = [[-1, 0],
#                [-0.95, 1]]

#BETTER THAN ANYTHING COUNTER
# num_strategies=2
# eta_1 =0.5
# eta_2 =0.1
# costs_table = [
#     [-1, -1],
#     [-0.99, 0.1]
# ]

#PRISONER DILEMMA COUNTER
# num_strategies=2
# eta_1 = 0.5
# eta_2 = 0.25
# costs_table = [[0.75, -1], [1, -0.9]]

#ONE ENTRY PROVEIMAGE
# num_strategies=2
# eta_1 = 0.5
# eta_2 = 0.25
# costs_table = [[-1, 0], [0, 0]]

#ONE ENTRY COUNTER
# num_strategies=2
# eta_1 = 0.5
# eta_2 = 0.25
# costs_table = [[1, 0], [0, 0]]

#ONE ENTRY PROVEIMAGE
# num_strategies=2
# eta_1 = 0.5
# eta_2 = 0.25
# costs_table = [[0, -1], [0, 0]]


#This code snippet then generates the plots
# eta_1 = 0.5
# cost_list1, weights_list1, cost_complete_list1 = analyse(100, eta_1, eta_1, costs_table, "ana_teste1")
# cost_list2, weights_list2, cost_complete_list2 = analyse(100, eta_1, eta_2, costs_table, "ana_teste2")
# visualize_diff((cost_list1, weights_list1, cost_complete_list1), (cost_list2, weights_list2, cost_complete_list2))

# This can be used to search for counterexamples
# num_strategies = 2
# for _ in range(100): #number of games searched
#     costs_table, eta_1, eta_2  = generate_costs_table()
#     simulate(100, eta_1, eta_2, costs_table)
