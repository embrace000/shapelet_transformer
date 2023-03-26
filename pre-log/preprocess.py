import re
import os
import numpy as np
import pandas as pd
import pickle

def call_extract (log):
    # 定义正则表达式模式，用于匹配函数调用行
    regex_pattern = r'<(.+):(.+)\s(.+)\((.*)\)> -> <(.+):(.+)\s(.+)\((.*)\)>'
    call_sequence=[]
    input_file = open(log, 'r', encoding='utf-8')
    for line in input_file:
        # 使用正则表达式模式匹配行
        match = re.match(regex_pattern, line.strip())
        # 如果匹配成功，提取函数调用的来源和目标
        if match:
            source_class = match.group(1)
            source_method_type = match.group(2)
            source_method = match.group(3)
            source_args = match.group(4)
            target_class = match.group(5)
            target_method_type = match.group(6)
            target_method = match.group(7)
            target_args = match.group(8)
            call_sequence.append(target_method)
    return call_sequence

if __name__ == '__main__':
    log_dir = os.path.abspath("d:\\数据集\\test_mal2016")
    log_files = os.listdir(log_dir)
    call_dict = {}
    with open('call_dict.pkl', 'rb') as f:
        call_dict = pickle.load(f)
    num_seq = []



    for  file in (log_files):
        call_sequence=call_extract (os.path.join(log_dir, file))

        for func in call_sequence:
            if func not in call_dict:
                call_dict[func] = len(call_dict)
            num_seq.append(call_dict[func])


    seq_length=20
    N=int(len(num_seq)/20)
    num_seq=num_seq[0:N*20]
    num_seq = np.array(num_seq)
    num_seq= num_seq.reshape((N, 20))

    data = pd.DataFrame(num_seq )
    data.to_csv('../data_sequence/sequence2.txt', sep=' ', index=False, header=False)

    with open('call_dict.pkl', 'wb') as f:
        pickle.dump(call_dict, f)

    with open('call_dict.txt', 'w', encoding='utf-8') as f:
        for key, value in call_dict.items():
            f.write('%s:%s\n' % (value, key))
