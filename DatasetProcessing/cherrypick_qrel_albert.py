
import json

evaluation_path = 'ALBERT/AnReu_albert-for-math-ar-base-ft' #'Full/colbertmath4_albert-albert-base-v2_AnReu-albert-for-math-ar-base-ft'
qrelfile_parent = './ARQMathAgg/dataset_v2/qrel_test'
aggregate_path = 'ARQMathAgg/dataset_v2/aggregates/collection_agg_test_extended.json'

eval_res_out_path = f'./Evaluation/{evaluation_path}'
runfile = f'{eval_res_out_path}/run'
qrelfile_cherrypicked = f'{eval_res_out_path}/qrel_cherrypicked'

selected_pids = []

with open(aggregate_path, 'r', encoding='utf-8') as file:
    json_array = json.load(file)

for obj in json_array:
    selected_pid = obj['pids'][obj['corr_idx']]
    selected_pids.append(selected_pid)

considered_qids = set()
with open(runfile, 'r', encoding='utf-8') as file:
    for line in file:
        qid = int(line.split(" ")[0])
        considered_qids.add(qid)

with open(qrelfile_cherrypicked, 'w', encoding='utf-8') as qrelfile:
    with open(qrelfile_parent, 'r', encoding='utf-8') as file:
        for line in file:
            qid = int(line.split(" ")[0])
            correct_pid = int(line.split(" ")[2])

            if(qid in considered_qids and correct_pid in selected_pids):
                qrelfile.write(line)

print(f"Cherrypicked qrel file for [{evaluation_path}]: {len(considered_qids)} queries")