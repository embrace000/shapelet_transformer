import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import lcs
from  tqdm import tqdm
from tensorflow import keras
from transformer_build.transformer import build_model
import numpy as np
import pandas as pd
import keras_metrics as km

if __name__ == '__main__':

  embedd_dim = 5
  def embedding_tranform1(data):
        # save_unit=np.negative(np.ones(len(data))).astype(int)
        window_size = 40
        save_shapelet = []
        seq_count = 0
        # save_unit=[[]]*len(data)
        # int(len(data) / 400)
        save_unit = [[]] * 80000
        for i in tqdm(range(2000)):
            l1 = data[i * window_size:(i + 1) * window_size]
            # 400个找一次
            for j in range(len(shapelets)):
                # print(j)
                l2 = shapelets[j]
                l1_init = 0
                l1_index = 0
                while 1:
                    length_, lists = lcs.calculate_LCS(l1[l1_init:window_size], l2)
                    if len(lists) != 5:
                        break
                    else:  # 如果匹配到了
                        # 从0开始匹配
                        for k in range(5):  # 开始找shapelet的位置    j 就是shapelet 号  k是 shapelet的调用号
                            temp = np.where(l1[l1_index:window_size] == shapelets[j][k])[0]  # 找起始位置
                            save_shapelet = save_shapelet + [l1_index + temp[0] + i * window_size]
                            l1_index = l1_index + temp[0] + 1  # 从下个位置开始匹配
                        for l in range(5):
                            save_unit[save_shapelet[l]] = save_unit[save_shapelet[l]] + [j]
                        save_shapelet = []
                        l1_init = l1_index
        return save_unit


  def embedding_tranform2(data_emb,data):
        emb_temp = []
        emb_temp1 = []
        emb_temp2 = []
        emb_result = []
        for i in tqdm(range(len(data_emb))):
            if len(data_emb[i]) > 0:
                # emb_temp1 = syscall_emb[data[i]]
                for j in range(len(data_emb[i])):
                    if j == 0:
                        emb_temp2 = unit_emb[data_emb[i][j]]
                    else:
                        emb_temp2 = np.vstack((emb_temp2, unit_emb[data_emb[i][j]]))
                if j > 0:
                    emb_temp2 = emb_temp2.mean(axis=0)

                # emb_temp = np.hstack((emb_temp1, emb_temp2))
                emb_temp = emb_temp2
                if len(emb_result) == 0:
                    emb_result = emb_temp
                else:
                    emb_result = np.vstack((emb_result, emb_temp))

        return emb_result


  def embedding_tranform(data):
        data_emb = embedding_tranform1(data)
        emb_result = embedding_tranform2(data_emb,data)
        return emb_result


  def vec_group(data_emb):
        seq_length = 20
        sequence = np.ones([1, seq_length, embedd_dim])
        for i in range(int(len(data_emb) / seq_length)):
            index = i * seq_length
            emb_temp = data_emb[index:index + seq_length, :]
            emb_temp = emb_temp.reshape(1, seq_length, embedd_dim)
            sequence = np.vstack((sequence, emb_temp))  # sequence
        sequence = np.delete(sequence, 0, axis=0)
        return sequence

################################################################

  # syscall_emb = pd.read_csv('call_unit_emb/call_emb.csv', index_col=0)
  # syscall_emb = np.array(syscall_emb)
  unit_emb = pd.read_csv('unit_dim.csv', index_col=0)
  unit_emb  = np.array(unit_emb )

  shapelets = np.loadtxt('shapelet_generation/shapelets.txt',dtype=int,delimiter=' ')

  data1=np.loadtxt('data_sequence/sequence1.txt',dtype=int,delimiter=' ')
  data1=data1.reshape(len(data1)*20)
  # data1 = data1[0:int(len(data1) / 2)]

  data2 = np.loadtxt('data_sequence/sequence2.txt', dtype=int, delimiter=' ')
  data2 = data2.reshape(len(data2) * 20)

  data3 = np.loadtxt('data_sequence/sequence3.txt', dtype=int, delimiter=' ')
  data3 = data3.reshape(len(data3) * 20)
  # data3 = data3[int(len(data3) / 2):len(data3)]

  data4 = np.loadtxt('data_sequence/sequence4.txt', dtype=int, delimiter=' ')
  data4 = data4.reshape(len(data4) * 20)

######################################################################
  data1_emb= embedding_tranform(data1)
  sequence1=vec_group(data1_emb)
  label1=np.zeros(len(sequence1))

  data2_emb = embedding_tranform(data2)
  sequence2 = vec_group(data2_emb)
  label2 = np.ones(len(sequence2))

  sequence = np.vstack((sequence1, sequence2))
  label=np.hstack((label1, label2))

  # data3_emb = embedding_tranform(data3)
  # sequence3 = vec_group(data3_emb)

  # data3_emb = embedding_tranform(data3)
  # sequence3 = vec_group(data3_emb)
  # label3 = np.zeros(len(sequence3))
  #
  # data4_emb = embedding_tranform(data4)
  # sequence4 = vec_group(data4_emb)
  # label4 = np.ones(len(sequence4))
  #
  # sequence_test = np.vstack((sequence3, sequence4))
  # label_test = np.hstack((label3, label4))



input_shape = sequence.shape[1:]
sequence_length=20
n_classes=2
# embedd_dim=768
model = build_model(
    input_shape,
    head_size=32,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.5,
    dropout=0.25,
    sequence_length=sequence_length,
    embedd_dim=embedd_dim,
    n_classes=2
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy",km.sparse_categorical_recall(),km.sparse_categorical_precision(),km.sparse_categorical_f1_score()],
)
#model.summary()

history=model.fit(
    sequence,
    label,
    validation_split=0.2,
    epochs=10000,
    batch_size=128,
    shuffle=False,
    # validation_data=(sequence_test, sequence_test)
)


# np.savetxt('wi-f1socre-0.2.csv', history.history['val_f1_score'])
# np.savetxt('wi-recall-0.2.csv', history.history['val_recall'])
# np.savetxt('wi-precision-0.2.csv', history.history['val_precision'])



model.save('my_mode.h5')
