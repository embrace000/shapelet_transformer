from tensorflow import keras
from transformer_build.transformer import build_model
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import keras_metrics as km
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from attention import Attention

###############################################
# emb_dim = pd.read_csv('emb_dim.csv', index_col=0)
# emb_dim = np.array(emb_dim)
# label = pd.read_csv('label.csv', index_col=0)
# label = np.array(label)
# sequence_length=50
# sequence_data=np.ones([1,seq_length,400])
# sequence_label=[0]
#
# for i in range(int(len(emb_dim)/seq_length)) :
#     index=i*seq_length
#     emb_temp= emb_dim[index:index+seq_length, :]
#     emb_temp=  emb_temp.reshape(1, seq_length, 400)
#     sequence_data=np.vstack((sequence_data,emb_temp)) # sequence
#     if (1 in label[index:index+seq_length]) :
#         sequence_label=np.hstack(( sequence_label,1))
#     else :
#         sequence_label = np.hstack((sequence_label, 0))
#
# sequence_data = np.delete(sequence_data, 0, axis=0)
# sequence_label = np.delete(sequence_label, 0, axis=0)

# data1 = np.loadtxt('transform1.txt', dtype=int, delimiter=' ')
# data2 = np.loadtxt('transform2.txt', dtype=int, delimiter=' ')
# data3 = np.loadtxt('transform3.txt', dtype=int, delimiter=' ')
#
# seq=20
#
# batch=int(len(data1)/seq)
# data1=data1[0:batch*seq]
# data1=np.reshape(data1,(batch,seq,1))
# label1=np.zeros(len(data1))
#
# batch=int(len(data2)/seq)
# data2=data2[0:batch*seq]
# data2=np.reshape(data2,(batch,seq,1))
# label2=np.ones(len(data2))
#
#
# batch=int(len(data2)/seq)
# data3=data3[0:batch*seq]
# data3=np.reshape(data3,(batch,seq,1))
# label3=np.ones(len(data3))
#
# sequence_data=np.vstack((data1,data2))
# sequence_label=np.hstack((label1,label2))
#
# sequence_data=np.vstack((sequence_data,data3))
# sequence_label=np.hstack((sequence_label,label3))
#################################################
data1 = np.loadtxt('../shapelet_generation/shapelets-30.txt', dtype=int, delimiter=' ')
data2 = np.loadtxt('../shapelet_generation/shapelets_label-30.txt', dtype=int, delimiter=' ')
seq=5
batch=len(data1)
sequence_data=np.reshape(data1,(batch,seq,1))
sequence_label=data2
#################################################

input_shape = sequence_data.shape[1:]

sequence_length=seq
embedd_dim=1
# model = build_model(
#     input_shape,
#     head_size=256,
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
model = build_model(
    input_shape,
    head_size=8,
    num_heads=1,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[32],
    mlp_dropout=0.2,
    dropout=0.2,
    sequence_length=sequence_length,
    embedd_dim=embedd_dim,
    n_classes=2
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["sparse_categorical_accuracy",km.sparse_categorical_recall(),km.sparse_categorical_precision(),km.sparse_categorical_f1_score()],
)
model.summary()

#callbacks = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
#callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

#checkpoint = ModelCheckpoint('{epoch}_model.h5',monitor='val_sparse_categorical_accuracy',save_best_only=True, period=1)

model.fit(
    sequence_data,
    sequence_label,
    # validation_split=0.3,
    epochs=10000,
    batch_size=128,
    shuffle=False,
    # validation_freq = 20
   # callbacks=[checkpoint],
)

model.save('my_mode.h5')
#
