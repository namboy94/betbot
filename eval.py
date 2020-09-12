import os
import json
import sys

parent = sys.argv[1]
stats = {}
for _file in os.listdir(parent):
    path = os.path.join(parent, _file)
    if _file.endswith(".json"):
        name = _file.split(".json")[0]
        with open(path, "r") as f:
            data = json.load(f)

        for key, val in data.items():
            if key not in stats:
                stats[key] = {}
            stats[key][name] = val


best = {}
for key, data in stats.items():
    print(key)
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_data:
        print(f"    {name}: {value}")
    best[key] = [sorted_data[0]]

print("Best:")
best_combis = []
for key, values in best.items():
    print(key)
    for name, value in values:
        print(f"    {name}: {value}")
        best_combis.append(name)
