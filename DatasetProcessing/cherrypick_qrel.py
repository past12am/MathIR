
evaluation_path = 'Full/colbertmath4_albert-albert-base-v2_AnReu-albert-for-math-ar-base-ft'
qrelfile_parent = './ARQMathAgg/dataset_v2/qrel_test'

eval_res_out_path = f'./Evaluation/{evaluation_path}'
runfile = f'{eval_res_out_path}/run'
qrelfile_cherrypicked = f'{eval_res_out_path}/qrel_cherrypicked'


considered_qids = set()
with open(runfile, 'r', encoding='utf-8') as file:
    for line in file:
        qid = int(line.split(" ")[0])
        considered_qids.add(qid)


with open(qrelfile_cherrypicked, 'w', encoding='utf-8') as qrelfile:
    with open(qrelfile_parent, 'r', encoding='utf-8') as file:
        for line in file:
            qid = int(line.split(" ")[0])

            if(qid in considered_qids):
                qrelfile.write(line)

print(f"Cherrypicked qrel file for [{evaluation_path}]: {len(considered_qids)} queries")