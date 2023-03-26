import os
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict
from sklearn.neighbors import NearestNeighbors, LSHForest
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)

params = vars(parser.parse_args())

eval_name = f'bgl_{params["train_anomaly_ratio"]}_tar'
seed = 42
data_dir = "../data/processed/BGL"
np.random.seed(seed)

params = {
    "log_file": "../data/BGL/BGL.log_structured.csv",
    "time_range": 1,  # 6 hours
    "train_ratio": None,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)


def load_BGL(
    log_file,
    time_range,
    train_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
):

    data1 = np.loadtxt( "../data/BGL/sequence1.txt", dtype=int, delimiter=' ')
    data1 = data1.reshape(len(data1) * 20)
    data2 = np.loadtxt("../data/BGL/sequence2.txt", dtype=int, delimiter=' ')
    data2 = data2.reshape(len(data2) * 20)
    data3 = np.loadtxt("../data/BGL/sequence3.txt", dtype=int, delimiter=' ')
    data3 = data3.reshape(len(data3) * 20)
    data4 = np.loadtxt("../data/BGL/sequence4-0.35.txt", dtype=int, delimiter=' ')
    # data4 = data4.reshape(len(data4) * 20)
    # data2=data4[0:int(len(data4)/2)]
    # data4 =data4[int(len(data4) / 2):len(data4)]
    #data4=data4[0:2000]
    #####################

    with open('../defense_new/g-seq-1.txt', 'r') as f:
        lines = f.readlines()
    data5 = []
    for line in tqdm(lines):
        row = [int(x) for x in line.split()]
        row = np.array(row)
        data5 = np.hstack((data5, row)).astype(int)
    data5 = data5[0:int(len(data5) / 20) * 20]
    data5 = np.reshape(data5, (int(len(data5) / 20), 20))

    with open('../defense_new/g-seq-2.txt', 'r') as f:
        lines = f.readlines()
    data6 = []
    for line in tqdm(lines):
        row = [int(x) for x in line.split()]
        row = np.array(row)
        data6 = np.hstack((data6, row)).astype(int)
    data6 = data6[0:int(len(data6) / 20) * 20]
    data6 = np.reshape(data6, (int(len(data6) / 20), 20))

    lshf1 = LSHForest(n_estimators=20, random_state=42)
    lshf1.fit(data5)

    lshf2 = LSHForest(n_estimators=20, random_state=42)
    lshf2.fit(data6)

    for j in tqdm(range(len(data4))):
        query_matrix = data4[j]  # 查询矩阵
        # 查询距离最近的点
        dist1, ind1 = lshf1.kneighbors([query_matrix.flatten()], n_neighbors=1)
        dist2, ind2 = lshf2.kneighbors([query_matrix.flatten()], n_neighbors=1)
        if dist1[0][0] <=dist2[0][0]:
            data4[j]=data5[ind1[0][0]]
        else:
            data4[j] = data6[ind2[0][0]]
    data4=np.reshape(data4,(20*len(data4)))



    session_dict = OrderedDict()
    sessid=0
    for idx, row in enumerate(data1):
        if idx == 0:
            sessid = 0
        elif idx % 20 ==0:
            sessid=sessid+1
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row)
        session_dict[sessid]["label"].append(0)

    for k, v in session_dict.items():
        session_dict[k]["label"] = [int(1 in v["label"])]
    session_idx = list(range(len(session_dict)))

    sessid = len(session_dict)
    for idx, row in enumerate(data2):
        if idx == 0:
            sessid = sessid
        elif idx % 20 == 0:
            sessid = sessid + 1
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row)
        session_dict[sessid]["label"].append(1)

    for k, v in session_dict.items():
        session_dict[k]["label"] = [int(1 in v["label"])]
    session_idx = list(range(len(session_dict)))

    session_dict_test = OrderedDict()
    sessid_test = 0
    for idx, row in enumerate(data3):
        if idx == 0:
            sessid_test = 0
        elif idx % 20 == 0:
            sessid_test = sessid_test + 1
        if sessid_test not in session_dict_test:
            session_dict_test[sessid_test] = defaultdict(list)
        session_dict_test[sessid_test]["templates"].append(row)
        session_dict_test[sessid_test]["label"].append(0)

    for k, v in session_dict_test.items():
        session_dict_test[k]["label"] = [int(1 in v["label"])]
    session_idx_test = list(range(len(session_dict_test)))

    sessid_test =  len(session_dict_test)
    for idx, row in enumerate(data4):
        if idx == 0:
            sessid_test = sessid_test
        elif idx % 20 == 0:
            sessid_test = sessid_test + 1
        if sessid_test not in session_dict_test:
            session_dict_test[sessid_test] = defaultdict(list)
        session_dict_test[sessid_test]["templates"].append(row)
        session_dict_test[sessid_test]["label"].append(1)
    for k, v in session_dict_test.items():
        session_dict_test[k]["label"] = [int(1 in v["label"])]
    session_idx_test = list(range(len(session_dict_test)))




    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)
    session_ids = np.array(list(session_dict.keys()))
    session_ids_test = np.array(list(session_dict_test.keys()))


    train_lines = len(session_idx)
    test_lines = len(session_idx_test)


    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx_test[0:test_lines]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids_test[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)+len(session_ids_test)))

    session_train = { k: session_dict[k] for k in session_id_train }
    session_test = {k: session_dict_test[k] for k in session_id_test}

    session_labels_train = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_train.items()
    ]
    session_labels_test = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_test.items()
    ]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))
    return session_train, session_test



if __name__ == "__main__":
    load_BGL(**params)
