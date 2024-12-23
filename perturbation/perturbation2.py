import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import random

def seq_generate(line) :
    temp1 = np.array([])
    line= line.split(' ')
    for i in range(len(line)):
            vectemp = line[i]
            vectemp= int(re.sub("\D", "", vectemp))

            temp1 = np.append( temp1,vectemp).astype(int)

    #         emb2 = embedding.one_embedding(vectemp, 400)
    #         emb_dim = np.vstack((emb_dim, emb2))
    # emb_dim = np.delete(emb_dim, 0, axis=0)
    return   temp1

def do_word():
    perd_length=0
    seq_lenth=len(df1)
    area_num=int(seq_lenth/per_num)
    for i in range (per_num) :
        j = random.randint(i*area_num+perd_length,(i+1)*area_num+perd_length)
        df2 = pd.DataFrame(seq_generate(str(line[random.randint(1, len(line))])))
        df1 = pd.DataFrame(np.insert(df1.values, j, df2, axis=0))
        perd_length=perd_length+len(df2)
    
    
    df1.to_csv('emb_adversical.csv')
    
    
    for m in range (len(df1)):
            label = np.hstack((label, 1))
    label = np.delete(label, 0, axis=0)
    data2 = pd.DataFrame(label)
    data2.to_csv('label_adversical.csv')

if __name__ == '__main__':


    f = open('test.txt', 'r')
    line = f.readlines()
    data1 = np.loadtxt('../data_sequence/sequence2.txt', dtype=int, delimiter=' ')
    data1 = data1.reshape(len(data1) * 20)
    # per_num=1600*8
    per_num=225*7
    perd_length = 0
    seq_lenth = len(data1)
    area_num = int(seq_lenth / per_num)
    for i in range(per_num):
        j = random.randint(i * area_num + perd_length, (i + 1) * area_num + perd_length)
        insert_seq = seq_generate(str(line[random.randint(1, len(line))]))
        data1=np.insert(data1, j,  insert_seq, axis=0)
        perd_length = perd_length + len(insert_seq)

    seq=20
    batch=int(len(data1)/seq)
    data1=data1[0:batch*seq]
    data1=np.reshape(data1,(batch,seq))
    data5 = pd.DataFrame(data1)
    data5.to_csv('../data_sequence/sequence3-0.35.txt', sep=' ', index=False, header=False)

    # perd_length=0
    # seq_lenth=len(df1)
    # area_num=int(seq_lenth/per_num)
    # for i in range (per_num) :
    #     j = random.randint(i*area_num+perd_length,(i+1)*area_num+perd_length)
    #     df2 = pd.DataFrame(seq_generate(str(line[random.randint(1, len(line))])))
    #     df1 = pd.DataFrame(np.insert(df1.values, j, df2, axis=0))
    #     perd_length=perd_length+len(df2)
    #
    #
    # df1.to_csv('emb_adversical.csv')
    #
    #
    # for m in range (len(df1)):
    #         label = np.hstack((label, 1))
    # label = np.delete(label, 0, axis=0)
    # data2 = pd.DataFrame(label)
    # data2.to_csv('label_adversical.csv')





  # df2 = pd.read_csv('emb_dim4.csv',index_col=0)
  # for i in range ( 200) :
  #     j=random.randint(1,9990)
  #     k=random.randint(1,9990)
  #     df = pd.DataFrame(np.insert(df1.values, j, df2.iloc[k:k+10], axis=0))
  #     df1=df
  #
  # df.to_csv('emb_adversical.csv')
  #
  # label=[0]  #记录label
  # for i in range(9998+2000) :
  #   label = np.hstack((label, 1))
  #
  # data2 = pd.DataFrame(label)
  # data2.to_csv('label_adversical.csv')


# label = np.hstack((label, 0))
        # print(line_new)


    # line1 = str(line[random.randint(1, len(line))])
    # m=seq_generate(line1)
    # m = seq_generate(line1)
    # m = seq_generate(line1)
    # m=seq_generate(str(line[random.randint(1, len(line))]))
    # label = np.delete(label, 0, axis=0)
    # emb_dim = np.delete(emb_dim, 0, axis=0)
    # data1 = pd.DataFrame(emb_dim)
    # data1.to_csv('emb_dim2.csv')
    # data2 = pd.DataFrame(label)
    # data2.to_csv('label_adversical.csv')
