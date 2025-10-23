# Let's implement Edmonds-Karp for the described logistics network and compute:
# 1) The max flow value
# 2) Edge flows
# 3) Aggregated Terminal -> Shop flows in a table
# 4) List of augmenting paths with bottlenecks (for step-by-step explanation)
#
# We'll also save the Terminal->Shop table to CSV and a concise Markdown report.

from collections import deque, defaultdict
import pandas as pd
import textwrap

# Define nodes
T1, T2 = "Термінал 1", "Термінал 2"
W1, W2, W3, W4 = "Склад 1", "Склад 2", "Склад 3", "Склад 4"
shops = [f"Магазин {i}" for i in range(1, 15)]
source, sink = "Джерело", "Сток"

nodes = [source, sink, T1, T2, W1, W2, W3, W4] + shops

# Directed edges with capacities (u -> v : capacity)
cap = defaultdict(lambda: defaultdict(int))

# Source to terminals (set to sum of their outgoing capacities so source isn't the bottleneck)
# We'll compute outgoing sums first
out_t1 = 25 + 20 + 15
out_t2 = 15 + 30 + 10

cap[source][T1] = out_t1
cap[source][T2] = out_t2

# Terminals to warehouses
cap[T1][W1] = 25
cap[T1][W2] = 20
cap[T1][W3] = 15

cap[T2][W3] = 15
cap[T2][W4] = 30
cap[T2][W2] = 10

# Warehouses to shops
cap[W1]["Магазин 1"] = 15
cap[W1]["Магазин 2"] = 10
cap[W1]["Магазин 3"] = 20

cap[W2]["Магазин 4"] = 15
cap[W2]["Магазин 5"] = 10
cap[W2]["Магазин 6"] = 25

cap[W3]["Магазин 7"] = 20
cap[W3]["Магазин 8"] = 15
cap[W3]["Магазин 9"] = 10

cap[W4]["Магазин 10"] = 20
cap[W4]["Магазин 11"] = 10
cap[W4]["Магазин 12"] = 15
cap[W4]["Магазин 13"] = 5
cap[W4]["Магазин 14"] = 10

# Shops to sink (give them high capacity so only the incoming edge limits)
for s in shops:
    # Large capacity (sum of all incoming to shops is <= 160), set to 10**9 to be safe
    cap[s][sink] = 10**9

# Build adjacency list for BFS (include reverse edges)
adj = defaultdict(list)
for u in list(cap.keys()):
    for v in list(cap[u].keys()):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

# Flow dictionary
flow = defaultdict(lambda: defaultdict(int))

augmentations = []  # to store (path, bottleneck)

def bfs_find_path(s, t):
    parent = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            residual = cap[u][v] - flow[u][v]
            if residual > 0 and v not in parent:
                parent[v] = u
                if v == t:
                    # Reconstruct path
                    path = []
                    cur = t
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    # Compute bottleneck
                    bottleneck = float('inf')
                    for i in range(len(path)-1):
                        a, b = path[i], path[i+1]
                        bottleneck = min(bottleneck, cap[a][b] - flow[a][b])
                    return path, bottleneck
                q.append(v)
    return None, 0

def edmonds_karp(s, t):
    maxflow = 0
    while True:
        path, bottleneck = bfs_find_path(s, t)
        if not path:
            break
        # augment along path
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck
        maxflow += bottleneck
        augmentations.append((path, bottleneck))
    return maxflow

max_flow_value = edmonds_karp(source, sink)

# Build Terminal -> Shop flow table
# For every shop, the incoming is from exactly one warehouse; the terminal is determined by which terminal sent flow to that warehouse through some path.
# We will trace contributions by decomposing actual flows along (Terminal -> Warehouse -> Shop) edges using the residual graph approach:
terminal_shop_flow = defaultdict(lambda: defaultdict(int))

# For each terminal and warehouse edge, we have flow[T][W]. For each warehouse->shop edge, we have flow[W][S].
# We need to split W->S flow by the proportions of T->W inflow.
# Compute total inflow to each warehouse from terminals
terminal_to_warehouse = defaultdict(lambda: defaultdict(int))
for T in [T1, T2]:
    for W in [W1, W2, W3, W4]:
        if flow[T][W] > 0:
            terminal_to_warehouse[T][W] = flow[T][W]

warehouse_in_total_from_T = defaultdict(int)
for W in [W1, W2, W3, W4]:
    warehouse_in_total_from_T[W] = sum(terminal_to_warehouse[T][W] for T in [T1, T2])

# Now distribute each W->S flow proportionally by terminal contributions
for W in [W1, W2, W3, W4]:
    w_to_shops = {s: flow[W][s] for s in shops if flow[W][s] > 0}
    total_in = warehouse_in_total_from_T[W]
    if total_in == 0:
        continue
    for s, f_ws in w_to_shops.items():
        for T in [T1, T2]:
            f_TW = terminal_to_warehouse[T][W]
            if f_TW == 0:
                continue
            share = f_ws * (f_TW / total_in)
            # Due to integrality, shares should be integers; but keep them as ints with rounding tolerance.
            terminal_shop_flow[T][s] += share

# Force integer rounding where tiny floating errors exist
for T in [T1, T2]:
    for s in shops:
        val = terminal_shop_flow[T][s]
        if abs(val - round(val)) < 1e-9:
            terminal_shop_flow[T][s] = int(round(val))

# Assemble DataFrame
rows = []
for T in [T1, T2]:
    for s in shops:
        rows.append({"Термінал": T, "Магазин": s, "Фактичний Потік (одиниць)": terminal_shop_flow[T][s] if terminal_shop_flow[T][s] else 0})
df = pd.DataFrame(rows)

# Sort for neatness
df["Магазин #"] = df["Магазин"].str.extract(r'(\d+)').astype(int)
df.sort_values(by=["Термінал", "Магазин #"], inplace=True)
df.drop(columns=["Магазин #"], inplace=True)

# Display table to the user
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Потоки між терміналами та магазинами (Едмондс—Карп)", df)

# Save CSV and a concise Markdown report
csv_path = "/mnt/data/terminal_to_shop_flows.csv"
df.to_csv(csv_path, index=False)

# Prepare edge flow summaries and bottlenecks
def nonzero_edge_flows():
    edges = []
    for u in cap:
        for v in cap[u]:
            if flow[u][v] > 0:
                edges.append((u, v, flow[u][v], cap[u][v]))
    return sorted(edges)

edge_list = nonzero_edge_flows()

aug_lines = []
for idx, (p, b) in enumerate(augmentations, 1):
    aug_lines.append(f"{idx:02d}. Шлях: " + " -> ".join(p) + f" | Приріст: {b}")

report_md = f"""# Звіт: Максимальний потік для логістичної мережі (Едмондс—Карп)

**Сумарний максимальний потік:** **{max_flow_value}** одиниць

## Ненульові потоки на ребрах (факт / місткість)
""" + "\n".join([f"- {u} → {v}: {f}/{c}" for (u,v,f,c) in edge_list]) + """

## Кроки алгоритму (послідовність доповнень)
""" + "\n".join(aug_lines) + """

## Підсумкова таблиця «Термінал → Магазин»
Збережено у файлі `terminal_to_shop_flows.csv`.
"""

md_path = "/mnt/data/report_maxflow_logistics.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(report_md)

csv_path, md_path, max_flow_value




from trie import Trie

class Homework(Trie):
    def count_words_with_suffix(self, pattern) -> int:
        # Перевірка типу вхідних даних
        if not isinstance(pattern, str):
            raise TypeError("Параметр pattern має бути рядком")
        if not pattern:
            return 0  # порожній шаблон не має сенсу

        # Метод перебирає всі слова в Trie (використовуючи існуючі ключі)
        count = 0
        for word in self.keys():
            if word.endswith(pattern):
                count += 1
        return count

    def has_prefix(self, prefix) -> bool:
        # Перевірка типу вхідних даних
        if not isinstance(prefix, str):
            raise TypeError("Параметр prefix має бути рядком")
        if not prefix:
            return False  # порожній префікс не вважаємо валідним

        # Проходимо по Trie, поки символи префікса співпадають
        node = self.root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        # Якщо дійшли сюди — існує хоча б одне слово з цим префіксом
        return True


if __name__ == "__main__":
    trie = Homework()
    words = ["apple", "application", "banana", "cat"]
    for i, word in enumerate(words):
        trie.put(word, i)

    # Перевірка кількості слів, що закінчуються на заданий суфікс
    assert trie.count_words_with_suffix("e") == 1      # apple
    assert trie.count_words_with_suffix("ion") == 1    # application
    assert trie.count_words_with_suffix("a") == 1      # banana
    assert trie.count_words_with_suffix("at") == 1     # cat

    # Перевірка наявності префікса
    assert trie.has_prefix("app") == True              # apple, application
    assert trie.has_prefix("bat") == False
    assert trie.has_prefix("ban") == True              # banana
    assert trie.has_prefix("ca") == True               # cat

    print("✅ Усі тести пройдено успішно!")
