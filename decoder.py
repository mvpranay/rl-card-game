import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--value_policy", type=str)
parser.add_argument("--testcase")

args_namespace = parser.parse_args()
value_policy_file = args_namespace.value_policy
testcase_file = args_namespace.testcase

if not os.path.isfile(value_policy_file):
    print("value policy file does not exist")
    exit(1)

if not os.path.isfile(testcase_file):
    print("testcase file does not exist")
    exit(1)

with open(testcase_file) as f:
    contents = f.read().split('\n')
threshold = int(contents[1])
hands = contents[5:]

optimal_actions = []
with open(value_policy_file) as f:
    contents = f.readlines()
    for line in contents:
        _, action = line.split()
        optimal_actions.append(int(action))

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

# process each hand and get optimal action
for hand in hands:
    hand_list = []
    suit = {}
    for card in hand.split():
        card_val = int(card[:-1])
        suit[card_val] = card[-1]
        hand_list.append(card_val)
    state_num = state_to_num[tuple(sorted(hand_list))]
    opt_action = optimal_actions[state_num]
    
    # if optimal action is to stop
    if opt_action == 14:
        print(27)
    # else if hit 
    elif opt_action == 0:
        print(0)
    # if swap, and currently heart
    elif suit[opt_action] == 'H':
        print(opt_action)
    # if swap, and currently diamond
    elif suit[opt_action] == 'D':
        print(opt_action + 13)
    else:
        print("you messed up bro")
