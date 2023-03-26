import numpy
import numpy as np
import pandas as pd
import lcs
from  tqdm import tqdm
from tensorflow import keras
from transformer_build.transformer import build_model
from transformer_build.transformer import PositionalEmbedding
import numpy as np
import pandas as pd
import keras_metrics as km
from keras.models import load_model

if __name__ == '__main__':

  def  embedding_tranform1(data) :
      # save_unit=np.negative(np.ones(len(data))).astype(int)
      window_size = 800
      save_shapelet=[]
      seq_count=0
      # save_unit=[[]]*len(data)
      #int(len(data) / 400)
      save_unit = [[]]*160000
      for i in tqdm(range(1000)):
           l1 = data[i * window_size:(i + 1) * window_size]
            #400个找一次
           for j in range(len(shapelets)):
                 # print(j)
                 l2 = shapelets[j]
                 l1_init = 0
                 l1_index = 0
                 while 1 :
                     length_, lists = lcs.calculate_LCS(l1[l1_init:window_size], l2)
                     if len(lists) != 5 :
                         break
                     else:  # 如果匹配到了
                        # 从0开始匹配
                        for k in range (5) :     #开始找shapelet的位置    j 就是shapelet 号  k是 shapelet的调用号
                            temp= np.where(l1[l1_index:window_size] == shapelets[j][k])[0]  #找起始位置
                            save_shapelet= save_shapelet+[l1_index+temp[0]+i*window_size]
                            l1_index=l1_index+temp[0]+1  #从下个位置开始匹配
                        for l in range(5):
                            save_unit[save_shapelet[l]]=save_unit[save_shapelet[l]]+[j]
                        save_shapelet = []
                        l1_init=l1_index
      return  save_unit


  def embedding_tranform2(data_emb,data):
      emb_temp = []
      emb_temp1 = []
      emb_temp2 = []
      emb_result = []
      for i in tqdm(range(len(data_emb))):
          if len(data_emb[i]) > 0:
              emb_temp1 = syscall_emb[data[i]]
              for j in range(len(data_emb[i])):
                  if j == 0:
                      emb_temp2 = unit_emb[data_emb[i][j]]
                  else:
                      emb_temp2 = np.vstack((emb_temp2, unit_emb[data_emb[i][j]]))
              if j > 0:
                  emb_temp2 = emb_temp2.mean(axis=0)

              emb_temp = np.hstack((emb_temp1, emb_temp2))
              emb_temp = emb_temp1
              if len(emb_result) == 0:
                  emb_result = emb_temp
              else:
                  emb_result = np.vstack((emb_result, emb_temp))

      return emb_result

  def embedding_tranform(data):
      data_emb=embedding_tranform1(data)
      emb_result=embedding_tranform2(data_emb,data)
      return emb_result


  def vec_group(data_emb):
      seq_length = 20
      sequence = np.ones([1, seq_length, 768])
      for i in range(int(len(data_emb) / seq_length)):
          index = i * seq_length
          emb_temp = data_emb[index:index + seq_length, :]
          emb_temp = emb_temp.reshape(1, seq_length, 768)
          sequence = np.vstack((sequence, emb_temp))  # sequence
      sequence = np.delete(sequence, 0, axis=0)
      return  sequence

################################################################

  syscall_emb = pd.read_csv('call_unit_emb/call_emb.csv', index_col=0)
  syscall_emb = np.array(syscall_emb)
  unit_emb = pd.read_csv('call_unit_emb/unit_dim-30.csv', index_col=0)
  unit_emb = np.array(unit_emb)

  shapelets = np.loadtxt('shapelet_generation/shapelets-30.txt', dtype=int, delimiter=' ')

  data1 = np.loadtxt('data_sequence/sequence1.txt', dtype=int, delimiter=' ')
  data1 = data1.reshape(len(data1) * 20)
  # data1 = data1[0:int(len(data1) / 2)]

  data2 = np.loadtxt('data_sequence/sequence2.txt', dtype=int, delimiter=' ')
  data2 = data2.reshape(len(data2) * 20)

  data3 = np.loadtxt('data_sequence/sequence1.txt', dtype=int, delimiter=' ')
  data3 = data3.reshape(len(data3) * 20)

  # data3 = data3[int(len(data3) / 2):len(data3)]

  data4 = np.loadtxt('data_sequence/sequence3-0.4.txt', dtype=int, delimiter=' ')
  data4 = data4.reshape(len(data4) * 20)

######################################################################
  data3_emb= embedding_tranform(data3)
  sequence3=vec_group(data3_emb)
  label3=np.zeros(len(sequence3))

  data4_emb = embedding_tranform(data4)
  sequence4 = vec_group(data4_emb)
  label4 = np.ones(len(sequence4))

  sequence = np.vstack((sequence3, sequence4))
  label=np.hstack((label3, label4))

  # data3_emb = embedding_tranform(data3)
  # sequence3 = vec_group(data3_emb)
  # label3 = np.zeros(len(sequence3))
  #
  # sequence = np.vstack((sequence, sequence3))
  # label = np.hstack((label, label3))
  #
  # data4_emb = embedding_tranform(data4)
  # sequence4 = vec_group(data4_emb)
  # label4 = np.ones(len(sequence4))
  #
  # sequence = np.vstack((sequence, sequence4))
  # label = np.hstack((label, label4))



input_shape = sequence.shape[1:]
sequence_length=20
n_classes=2
embedd_dim=768
# model = build_model(
#     input_shape,
#     head_size=32,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=4,
#     mlp_units=[128],
#     mlp_dropout=0.5,
#     dropout=0.25,
#     sequence_length=sequence_length,
#     embedd_dim=embedd_dim,
#     n_classes=2
# )

model= load_model("my_mode.h5",custom_objects={'sparse_categorical_recall': km.sparse_categorical_recall,\
                                               'sparse_categorical_precision':km.sparse_categorical_precision, \
                                               'sparse_categorical_f1_score': km.sparse_categorical_f1_score,  \
                                               'PositionalEmbedding':PositionalEmbedding})


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy",km.sparse_categorical_recall(),km.sparse_categorical_precision(),km.sparse_categorical_f1_score()],
)
#model.summary()
#
# model.fit(
#     sequence,
#     label,
#     validation_split=0.5,
#     epochs=2000,
#     batch_size=128,
#     shuffle=False,
# )
#


k=model.predict(
    sequence,
    # sequence_label,
    # validation_split=0.4,
    # epochs=500,
    batch_size=128,
    # shuffle=True,
   # callbacks=[checkpoint],
)

count=0
for i in range (len(k)) :
    if k[i][0]<0.5:
        count=count+1
m=k[:,1]

np.savetxt('roc-score-wo-0.4.csv', m)
np.savetxt('roc-label-wo-0.4.csv', label)

print(len(k))
print(count)












'''
  temp1=np.array([])
  temp2 = np.array([])
  temp3 = np.array([])
  temp4= np.array([])
  temp5 = np.array([])

  for i in tqdm (range(int(0.2*int(len(data1)/400)))) :
      l1 =data1 [i*400:(i+1)*400]

      for j in range(len(shapelets)):
         l2  =   shapelets[j]
         length_,lists = lcs.calculate_LCS(l1, l2)
         if len(lists)==5 :
            temp1 = np.append( temp1,j).astype(int)
            temp2 = np.append(temp2 ,np.where(l1 == shapelets[j][0])[0][0]).astype(int)
            temp_test = np.where(l1 == shapelets[j][0])[0]
            print(np.where(l1 == shapelets[j][0])[0])


      if len(temp1>0):
          index=np.argsort(temp2)
          posi_temp=  np.sort(temp2)
          for k in range(len(index)) :
              temp3=np.append(temp3, temp1[index[k]]).astype(int)
              temp4 = np.append(temp4,  posi_temp[k]).astype(int)
              # print(temp3)
      temp1 = np.array([])
      temp2 = np.array([])

  temp5 = numpy.diff(temp4)
  temp5 =  np.append( [0],temp5).astype(int)

  # 对temp3 序列 和 temp5序列差值进行操作 差值小于20认为重合 大于20认为不重合
  seqlen = 20 #input_size
  temp6= np.array([])
  temp7 = np.ones(seqlen)

  def bound_diff(index):
      if index > 20:
          return 0
      else:
          return 1

  for i in range(len(temp3)):
      if i == 0:
          temp6 = np.append(temp6, temp3[i])
      else:
          if bound_diff(temp5[i]) == bound_diff(temp5[i - 1]):
              temp6 = np.append(temp6, temp3[i])
          else:
              if len(temp6) < seqlen:    #补齐
                  temp6 = np.pad(temp6, (0, seqlen - len(temp6)), 'constant', constant_values=(0, -1))
              else:
                  temp6 = temp6[0:seqlen]   #截断
              #print(temp6)
              temp7 = np.vstack((temp7, temp6))
              temp6 = np.append([], temp3[i])

  temp7 = np.delete(temp7, 0, axis=0)
  data = pd.DataFrame(temp7)
  data.to_csv('transform3-0.4.txt', sep=' ', index=False, header=False)
'''


  # print(temp)
  # # get two lists
  # l1 = [5, 3, 4, 212, 7, 6, 1, 7, 256, 189, 7, 167, 5]
  # l2 = [212, 7, 256, 5]
  # lists = lcs.lcs(l1, l2)
  # for l in lists:
  #     print(l)

##########
  # l1 =np.array( [5, 3, 4, 212, 7, 6, 1, 7, 256, 189, 7, 167, 5])
  # m=np.where(l1==212)
  # m=m[0][0]
  # print(m)
