import itertools
from collections import Counter
from typing import List, Tuple, Set, Dict


# Task 1: Count of each set in the list of element sets
def count(set1: Tuple[str, ...], data: List[List[str]]) -> int:
    s1 = set(set1)
    return sum(1 for s in map(set, data) if s1.issubset(s))


# Task 2: Self-join operation on a set of items
def get_union(itemset: List[List[str]], length: int) -> List[Set[str]]:
    union_set = []
    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            t = set(itemset[i]).union(set(itemset[j]))
            if len(t) == length:
                union_set.append(list(t))
    return union_set


# Task 3: Generate nonempty subsets excluding the superset
def generate_nonempty_subsets(items: List[str]) -> List[Tuple[str, ...]]:
    subsets = []
    for r in range(1, len(items)):
        subsets.extend(itertools.combinations(items, r))
    return subsets


# Task B: Apriori Algorithm
def apriori_algorithm(data: List[List[str]], min_support: int, min_confidence: float):
    def get_frequent_itemsets(candidate_itemsets: List[Set[str]], min_support_count: int) -> Dict[frozenset, int]:
        itemset_count = Counter()
        for itemset in candidate_itemsets:
            itemset_count[itemset] = sum(1 for transaction in data if itemset.issubset(transaction))
        return {itemset: count for itemset, count in itemset_count.items() if count >= min_support_count}

    # Step 1: Generate C1
    items = sorted(set(item for transaction in data for item in transaction))
    c1 = [frozenset([item]) for item in items]
    l1 = get_frequent_itemsets(c1, min_support)
    frequent_itemsets = l1

    k = 1
    while True:
        # Step 2:
        c_k = get_union(list(frequent_itemsets.keys()), k + 1)
        l_k = get_frequent_itemsets(c_k, min_support)
        if not l_k:
            break
        frequent_itemsets = l_k
        k += 1

    # Generating Association Rules
    def generate_rules(frequent_itemsets: Dict[frozenset, int], min_confidence: float):
        rules = []
        for itemset, support_count in frequent_itemsets.items():
            subsets = generate_nonempty_subsets(list(itemset))
            for subset in subsets:
                subset = frozenset(subset)
                confidence = support_count / frequent_itemsets[subset]
                if confidence >= min_confidence:
                    rule = (subset, itemset - subset)
                    rules.append((rule, confidence))
        return rules

    rules = generate_rules(frequent_itemsets, min_confidence)
    return frequent_itemsets, rules


#  data and parameters
itemset = [['a', 'b', 'c', 'd'], ['b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e']]
print("Task 1:")
print(count(('a', 'b', 'c'), itemset))

print("\nTask 2:")
print(get_union([['a', 'b'], ['a', 'c'], ['b', 'c'], ['b', 'd'], ['c', 'd'], ['c', 'e'], ['c', 'f']], 3))

print("\nTask 3:")
print(generate_nonempty_subsets(['a', 'b', 'c']))

print("\nTask B:")
data = [
    set(["a", "b", "c"]), set(["a", "b"]), set(["a", "b", "d"]),
    set(["b", "e"]), set(["b", "c", "e"]), set(["a", "d", "e"]),
    set(["a", "c"]), set(["a", "b", "d"]), set(["c", "e"]),
    set(["a", "b", "d", "e"]), set(["a", "b", "e", "c"])
]
min_support = 3
min_confidence = 0.8

frequent_itemsets, rules = apriori_algorithm(data, min_support, min_confidence)

print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{set(itemset)}: {count}")

print("\nAssociation Rules:")
for rule, confidence in rules:
    antecedent, consequent = rule
    print(f"{set(antecedent)} -> {set(consequent)}: {confidence * 100:.2f}% confidence")
