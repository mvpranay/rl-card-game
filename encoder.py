import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--game_config', type=str)

args_namespace = parser.parse_args()
game_config_file = args_namespace.game_config

if not os.path.isfile(game_config_file):
    print("Game config file does not exist")
    exit(1)

with open(game_config_file) as f:
    contents = f.read().split('\n')

_, threshold, bonus_val, bonus_seq = contents
threshold = int(threshold)
bonus_val = int(bonus_val)
bonus_seq = [int(item) for item in bonus_seq.split()]

def bonus_applicable(curr_state):
    if bonus_seq[0] not in curr_state:
        return False
    if bonus_seq[1] not in curr_state:
        return False
    if bonus_seq[2] not in curr_state:
        return False
    return True

state_to_num = {}

def number_states():
    stack = [()]
    visited = set()
    visited.add(())

    while len(stack) > 0:
        curr = stack.pop()

        if curr in state_to_num:
            continue

        state_to_num[curr] = len(state_to_num)

        # hitting
        for new_card in range(1, 14):
            if curr.count(new_card) == 2:
                continue
            next_state = tuple(sorted(curr + (new_card,)))
            if sum(next_state) < threshold and next_state not in visited:
                visited.add(next_state)
                stack.append(next_state)

        # swapping
        for swap_index in range(len(curr)):
            for new_card in range(1, 14):
                if curr.count(new_card) == 2:
                    continue
                next_state = tuple(sorted(curr[:swap_index] + (new_card,) + curr[swap_index+1:]))
                if sum(next_state) < threshold and next_state not in visited:
                    visited.add(next_state)
                    stack.append(next_state)

    # exit state
    exit_num = len(state_to_num)
    state_to_num["exit"] = exit_num
    return exit_num

exit_num = number_states()
numStates = len(state_to_num)
numActions = 15

print("numStates", numStates)
print("numActions", numActions)
print("end", exit_num)

# print transitions
def print_transitions():
    stack = [()]

    visited = set()
    visited.add(())

    while len(stack) > 0:
        curr = stack.pop()
        curr_state_num = state_to_num[curr]

        # hitting
        bust_prob = 0.0
        for new_card in range(1, 14):
            if curr.count(new_card) == 2:
                continue
            next_state = tuple(sorted(curr + (new_card,)))
            transition_probability = (2.0 - curr.count(new_card)) / (26.0 - len(curr))
            # if next state valid
            if next_state in state_to_num:
                print("transition", curr_state_num, 0, state_to_num[next_state], 0, transition_probability)
                if next_state not in visited:
                    visited.add(next_state)
                    stack.append(next_state)
            # bust
            else:
                bust_prob += transition_probability
        # bust while hitting
        print("transition", curr_state_num, 0, exit_num, 0, bust_prob)

        # swapping
        prev_card = -1
        for swap_index in range(len(curr)):
            if curr[swap_index] == prev_card:
                continue
            for new_card in range(1, 14):
                if curr.count(new_card) == 2:
                    continue
                next_state = tuple(sorted(curr[:swap_index] + (new_card,) + curr[swap_index+1:]))
                
                transition_probability = (2.0 - curr.count(new_card)) / (26.0 - len(curr))

                # if next state valid
                if next_state in state_to_num:
                    print("transition", curr_state_num, curr[swap_index], state_to_num[next_state], 0, transition_probability)
                    if next_state not in visited:
                        visited.add(next_state)
                        stack.append(next_state)
                # bust while trying to swap that card
                else:
                    print("transition", curr_state_num, curr[swap_index], exit_num, 0, transition_probability)
            prev_card = curr[swap_index]
        
        # exit 
        final_reward = sum(curr)
        if bonus_applicable(curr):
            final_reward += bonus_val
        print("transition", curr_state_num, 14, exit_num, final_reward, 1.0)

print_transitions()

print("mdptype", "episodic")
print("discount", 1.0)
