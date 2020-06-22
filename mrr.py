import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Calculate recall')
parser.add_argument('--result_file', type=str, help='file contains predicted result', required=True)
parser.add_argument('--top_k', type=int, help='top k highest rank items', required=True)
args = parser.parse_args()
result_file = args.result_file
top_k = args.top_k
list_seq = []
list_seq_topk_predicted = []
with open(result_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if(i == 0):
            continue
        if(i % 2 == 1):
            ground_truth = line.split(':')[-1]
            list_item = ground_truth.split()
            list_seq.append(list_item.copy())
            list_item.clear()
        if(i % 2 == 0):
            predicted_items = line.split('|')[1:top_k+1]
            list_top_k_item = []
            for item in predicted_items:
                item_key = item.strip().split(':')[0]
                list_top_k_item.append(item_key)
            list_seq_topk_predicted.append(list_top_k_item.copy())
            list_top_k_item.clear()
list_mrr = []
# print("list seq:")
# print(list_seq)
# print(len(list_seq))
# print("list predicted items")
# print(list_seq_topk_predicted)
# print(len(list_seq_topk_predicted))
for i, ground_truth in enumerate(list_seq):
  correct = 0
  max_rank = top_k + 1
  # print(i)
  for item_idx, item in enumerate(list_seq_topk_predicted[i]):
    if (item in ground_truth):
        max_rank = item_idx
        break
  if (max_rank > top_k):
      list_mrr.append(0)
  else:
      list_mrr.append(float(1)/float(max_rank))

print(np.array(list_mrr).mean())