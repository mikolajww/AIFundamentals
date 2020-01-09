import math
import copy
import sys
import time

distances = \
    {\
        "A": {"A": 0,"B": 2,"C": 1,"S": 11},\
        "B": {"A": 2,"B": 0,"C": 4,"S": 5} ,\
        "C": {"A": 1,"B": 4,"C": 0,"S": 8} ,\
        "S": {"A": 11,"B": 5,"C": 8,"S": 0} \
    }
demand = {"A": 5,"B": 3, "C": 7}
truck_pos = "S"
truck_capacity = 4

class State:
    def __init__(self, items_delivered, truck_pos, truck_items, parent, cost):
        self.items_delivered = items_delivered
        self.truck_pos = truck_pos
        self.truck_items = truck_items
        self.parent = parent
        self.cost = cost
    
    def __str__(self):
        return f'||Truck|items={self.truck_items}|pos={self.truck_pos}||Demand={self.items_delivered}||Total cost={self.cost}||'

    def print_recursive(self):
        if(self.parent is not None):
            self.parent.print_recursive()
        print(self)

def deliver_to(current_demand, recipient, num):
    new_demand = copy.deepcopy(current_demand)
    new_demand[recipient] -= num
    return new_demand

def is_satisfied(current_demand):
    for recipient_items in current_demand.values():
        if(recipient_items > 0):
            return False
    return True

def total_demand(current_demand):
    _sum = 0
    for recipient_items in current_demand.values():
        _sum += recipient_items
    return _sum


def expand(state: State):
    new_states = []
    if(state.truck_items == 0):
        travel_cost = distances[state.truck_pos]["S"]
        new_state = State(state.items_delivered, "S", truck_capacity, state, state.cost + travel_cost)
        new_states.append(new_state)
        return new_states
    else:
        for recipient, recipient_items in state.items_delivered.items():
            if(state.truck_pos != recipient):
                for i in range(1, state.truck_items + 1):
                    if(i > recipient_items):
                        break
                    else:
                        after_delivery = deliver_to(state.items_delivered, recipient, i)
                        travel_cost = distances[state.truck_pos][recipient]
                        new_state = State(after_delivery, recipient, state.truck_items - i, state, state.cost + travel_cost)
                        new_states.append(new_state)
        return new_states

def find_next_best_state(list_of_states: [State], mode='heuristic'):    
    if(mode == 'bfs'):
        return 0
    elif(mode == 'dfs'):
        return -1
    else:
        current_best_state = math.inf
        current_best_index = 0
        for index, s in enumerate(list_of_states):
            total_sum = total_demand(s.items_delivered)
            if(total_sum < current_best_state):
                current_best_state = total_sum
                current_best_index = index
        return current_best_index

if(len(sys.argv) == 2):
    mode = sys.argv[1]
else:
    mode = 'heuristic'

start = time.perf_counter()
Q = []
initial_state = State(demand, truck_pos, truck_capacity, None, 0)
Q.append(initial_state)
while(True):
    idx = find_next_best_state(Q, mode=mode)
    s = Q.pop(idx)
    if(is_satisfied(s.items_delivered)):
        stop = time.perf_counter()
        print(f'Found solution! Elapsed time: {stop - start}')
        s.print_recursive()
        break
    tmp = expand(s)
    Q.extend(tmp)
    '''
    for t in Q:
        print(t)
    '''