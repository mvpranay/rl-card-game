import argparse
import os
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, PULP_CBC_CMD, LpStatus, value, lpSum

# return mdpmodel
def read_mdp(input_file_path):
    with open(input_file_path) as file:
        contents = file.readlines()

    num_states = int(contents[0].split()[1])
    num_actions = int(contents[1].split()[1])
    end_states = np.full(num_states, False)
    for end_state in contents[2].split()[1:]:
        end_state = int(end_state)
        if end_state == -1:
            break
        end_states[end_state] = True

    transition_matrix = np.zeros((num_states, num_actions, num_states), dtype=float)
    reward_matrix = np.zeros((num_states, num_actions, num_states), dtype=float)

    # stores states reachable from s, taking action a
    reachable = [[set() for _ in range(num_actions)] for _ in range(num_states)]

    mdptype = contents[-2].split()[1]
    discount_factor = float(contents[-1].split()[1])

    contents = contents[3:-2]

    for line in contents:
        _, s1, ac, s2, r, p = line.split()
        s1 = int(s1)
        ac = int(ac)
        s2 = int(s2)
        r = float(r)
        p = float(p)
        transition_matrix[s1][ac][s2] = p
        reward_matrix[s1][ac][s2] = r
        reachable[s1][ac].add(s2)

    return [transition_matrix, reward_matrix, end_states, discount_factor, mdptype, reachable]

def read_policy_file(file_path):
    with open(file_path) as file:
        contents = file.readlines()
    return np.array([int(policy) for policy in contents])

def evaluate_policy(mdpmodel, policy):
    trans, rew, end, gamma, mdptype, reachable = mdpmodel
    num_states, num_actions, _ = trans.shape

    prob = LpProblem("_", LpMinimize)
    # define decision variables, V(s)
    V = [0 for _ in range(num_states)]
    for s in range(num_states):
        if not end[s]:
            V[s] = LpVariable(f"V{s}")

    # add the bellman equations
    for s in range(num_states):
        rhs = 0
        a = policy[s]
        for _s in reachable[s][a]:
            rhs += trans[s][a][_s] * (rew[s][a][_s] + gamma * V[_s])
        prob += V[s] == rhs

    # solve the lp Problem
    prob.solve(PULP_CBC_CMD(msg=False))
    
    ans = []
    for var in V:
        if type(var) is not int:
            ans.append(var.value())
        else:
            ans.append(0.0)
    return ans

# for float comparison
threshold = 1e-6

def get_max_improved_policy(value, mdpmodel):
    trans, rew, end, gamma, mdptype, reachable = mdpmodel
    num_states, num_actions, _ = trans.shape

    improved_policy = [-1 for _ in range(num_states)]
    best_action_value = value.copy()

    for s in range(num_states):
        for a in range(num_actions):
            sum = 0.0
            for _s in reachable[s][a]:
                sum += trans[s][a][_s] * (rew[s][a][_s] + gamma * value[_s])
            
            if sum > best_action_value[s] + threshold:
                best_action_value[s] = sum
                improved_policy[s] = a
    
    return improved_policy

def solve_hpi(mdpmodel):
    trans, rew, end, gamma, mdptype, reachable = mdpmodel
    num_states, num_actions, _ = trans.shape

    curr_policy = [0 for _ in range(num_states)]
    curr_value = evaluate_policy(mdpmodel, curr_policy)

    improved_policy = get_max_improved_policy(curr_value, mdpmodel)
    no_improvement = [-1 for _ in range(num_states)]

    while improved_policy != no_improvement:
        for s in range(num_states):
            if improved_policy[s] == -1:
                continue

            curr_policy[s] = improved_policy[s]
        
        curr_value = evaluate_policy(mdpmodel, curr_policy)
        improved_policy = get_max_improved_policy(curr_value, mdpmodel)

    # print optimal value function, policy
    for s in range(num_states):
        print(f"{curr_value[s]:.6f} {curr_policy[s]}")
    return

def solve_lp(mdpmodel):
    trans, rew, end, gamma, mdptype, reachable = mdpmodel
    num_states, num_actions, _ = trans.shape
    
    # formulate the lp problem
    prob = LpProblem("LP_Problem", LpMinimize)

    # define decision variables, V(s)
    V = [LpVariable(f"V{i}") for i in range(num_states)]
    
    # add the objective function
    prob += lpSum(V[s] for s in range(num_states)), "Objective"

    # add constraints
    for s in range(num_states):
        if end[s]:
            prob += V[s] == 0
            continue
        for a in range(num_actions):
            rhs = 0.0
            for _s in reachable[s][a]:
                rhs += trans[s][a][_s] * (rew[s][a][_s] + gamma * V[_s])
            prob += V[s] >= rhs

    # solve the lp Problem
    prob.solve(PULP_CBC_CMD(msg=False))

    # calculate optimal policy
    V_values = np.array([v.value() for v in V])
    optimal_policy = [0 for _ in range(num_states)]

    for s in range(num_states):
        max_value = float('-inf')
        max_action = -1
        for a in range(num_actions):
            # calculate q value
            sum = 0.0
            for _s in reachable[s][a]:
                sum += trans[s][a][_s] * (rew[s][a][_s] + gamma * V_values[_s])
            if sum > max_value:
                max_value = sum
                max_action = a
        optimal_policy[s] = max_action

    # print the obtained values
    for s in range(num_states):
        print(f"{V[s].value():.6f} {int(optimal_policy[s])}")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--algorithm', type=str, default='lp')

    args_namespace = parser.parse_args()

    input_file = args_namespace.mdp
    policy_file = args_namespace.policy
    algo = args_namespace.algorithm

    if not os.path.isfile(input_file):
        print("MDP file does not exist")
        exit(1)
    
    findOptimal = False
    if policy_file is None:
        findOptimal = True
    elif not os.path.isfile(policy_file):
        print("Policy file does not exist")
        exit(1)

    if algo not in ['hpi', 'lp']:
        print("Invalid algo")
        exit(1)
    
    mdpmodel = read_mdp(input_file)
    trans, rew, end, gamma, mdptype, reachable = mdpmodel
    num_states, num_actions, _ = trans.shape
    
    if not findOptimal:
        policy = read_policy_file(policy_file)
        values = evaluate_policy(mdpmodel, policy)
        for s in range(num_states):
            print(f"{values[s]:.6f} {policy[s]}")
    else:
        if algo == 'lp':
            solve_lp(mdpmodel)
        elif algo == 'hpi':
            solve_hpi(mdpmodel)

if __name__ == '__main__':
    main()
