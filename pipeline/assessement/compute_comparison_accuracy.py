import sys
import json
online_file, chatgpt_file, comparison_file = sys.argv[1], sys.argv[2], sys.argv[3]

with open(online_file) as f:
    data = json.load(f)
    online_scores = [d["score"] for d in data]

with open(chatgpt_file) as f:
    data = json.load(f)
    chatgpt_scores = [d["score"] for d in data]

with open(comparison_file) as f:
    comparisons = []
    for line in f:
        comparisons.append(float(line.strip()))

correct, total = 0,0
for online_score, chatgpt_score, comparison in zip(online_scores,chatgpt_scores,comparisons):
    if online_score == chatgpt_score == 0:
        continue
    else:
        total += 1
        if online_score > chatgpt_score and comparison == 1:
            correct += 1
        elif online_score < chatgpt_score and comparison == 2:
            correct += 1


print(correct / total, total)
        