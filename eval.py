# import os
# import json
# import sys
#
# parent = sys.argv[1]
# stats = {}
# for _file in os.listdir(parent):
#     path = os.path.join(parent, _file)
#     if _file.endswith(".json"):
#         name = _file.split(".json")[0]
#         with open(path, "r") as f:
#             data = json.load(f)
#
#         for key, val in data.items():
#             if key not in stats:
#                 stats[key] = {}
#             stats[key][name] = val
#
#
# best = {}
# for key, data in stats.items():
#     print(key)
#     sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
#     for name, value in sorted_data:
#         print(f"    {name}: {value}")
#     best[key] = [sorted_data[0]]
#
# print("Best:")
# best_combis = []
# for key, values in best.items():
#     print(key)
#     for name, value in values:
#         print(f"    {name}: {value}")
#         best_combis.append(name)


import os
import sys

parent = sys.argv[1]
combi_stats = {}
for child in os.listdir(parent):
    path = os.path.join(parent, child)
    if not os.path.isdir(path):
        continue

    avg = float(child.split("[")[1].split("]")[0])
    best = float(child.split("[")[2].split("]")[0])
    identifier = path.split("]")[-1].strip()
    name, hidden_activation, output_activation, loss, optimizer = identifier.split("-")
    identifier = identifier.split("-", 1)[1]
    if identifier not in combi_stats:
        combi_stats[identifier] = []
    combi_stats[identifier].append((avg, best))

ranking = []
for combi, stats in combi_stats.items():
    averages = sum([x[0] for x in stats]) / len(stats)
    best = sum([x[1] for x in stats]) / len(stats)
    ranking.append((combi, averages, best))

ranking.sort(key=lambda x: x[1])

for name, avg, best in ranking:
    print(f"{name} {avg} {best}")
