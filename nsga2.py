import math
import random

def func1(x):
    return x**2

def func2(x):
    return (x - 2)**2

funcs = [func1, func2]
bound = (-1000, 1000)
pop_size = 50
mutation_rate = 0.1
crossover_rate = 0.9
max_iter = 500

def init_pop():
    return [random.uniform(bound[0], bound[1]) for _ in range(pop_size)]

def sbx_crossover(x1, x2, eta=15):
    if random.random() > crossover_rate:
        return x1, x2

    u = random.random()
    beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    
    child1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    child2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

    return max(bound[0], min(bound[1], child1)), max(bound[0], min(bound[1], child2))

def polynomial_mutation(x, eta=20):
    if random.random() > mutation_rate:
        return x

    delta = random.random()
    if delta < 0.5:
        delta_q = (2 * delta) ** (1 / (eta + 1)) - 1
    else:
        delta_q = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))

    mutated_x = x + delta_q * (bound[1] - bound[0])
    return max(bound[0], min(bound[1], mutated_x))

def make_new_pop(P):
    new_pop = []
    while len(new_pop) < len(P):
        p1, p2 = random.sample(P, 2)
        c1, c2 = sbx_crossover(p1, p2)
        new_pop.append(polynomial_mutation(c1))
        new_pop.append(polynomial_mutation(c2))
    return new_pop[:len(P)]

def dominate(val1, val2):
    return all(v1 <= v2 for v1, v2 in zip(val1, val2)) and val1 != val2

def fast_non_dominated_sort(P):
    values = [[func(p) for func in funcs] for p in P]
    S, n, rank = [[] for _ in range(len(P))], [0] * len(P), [0] * len(P)
    F = [[]]

    for p in range(len(P)):
        for q in range(len(P)):
            if dominate(values[p], values[q]):
                S[p].append(q)
            elif dominate(values[q], values[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            F[0].append(p)

    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        F.append(Q)
    
    del F[-1]
    return [[P[i] for i in f] for f in F]

def crowding_distance_assignment(F):
    l = len(F)
    if l == 0:
        return []

    distance = [0] * l
    for m in range(len(funcs)):
        values = [funcs[m](i) for i in F]
        sorted_idx = sorted(range(l), key=lambda k: values[k])
        distance[sorted_idx[0]] = float('inf')
        distance[sorted_idx[-1]] = float('inf')
        min_val, max_val = values[sorted_idx[0]], values[sorted_idx[-1]]

        if max_val == min_val:
            continue
        
        for i in range(1, l - 1):
            distance[sorted_idx[i]] += (values[sorted_idx[i + 1]] - values[sorted_idx[i - 1]]) / (max_val - min_val)

    return distance

def main():
    P = init_pop()
    t = 0

    while t < max_iter:
        Q = make_new_pop(P)
        R = P + Q
        F = fast_non_dominated_sort(R)

        P_next = []
        i = 0
        while len(P_next) + len(F[i]) <= pop_size:
            P_next += F[i]
            i += 1

        if len(P_next) < pop_size:
            distances = crowding_distance_assignment(F[i])
            sorted_indices = sorted(range(len(F[i])), key=lambda k: distances[k], reverse=True)
            P_next += [F[i][idx] for idx in sorted_indices[:pop_size - len(P_next)]]

        P = P_next
        t += 1

        if t % 50 == 0:
            print(f"Iteration {t}: Best Front = {F[0]}")

    print("Final Pareto Front:")
    for p in F[0]:
        print(f"x: {p:.4f}, f1: {func1(p):.4f}, f2: {func2(p):.4f}")

if __name__ == '__main__':
    main()
