import sys

scores = []
with open(sys.argv[1]) as f:
    for line in f:
        scores.append(float(line.strip()))

print(sum(scores) / len(scores))
