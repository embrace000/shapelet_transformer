from tensorflow import keras
from transformer_build.transformer import build_model
from transformer_build.transformer import PositionalEmbedding
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import keras_metrics as km
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from attention import Attention
import  tensorflow
###############################################

# data1 = np.loadtxt('transform1.txt', dtype=int, delimiter=' ')
#
# seq=20
# batch=int(len(data1)/seq)
# data1=data1[0:batch*seq]
# data1=np.reshape(data1,(batch,seq,1))
# label1=np.zeros(len(data1))

data1 = np.loadtxt('../shapelet_generation/shapelets-30.txt', dtype=int, delimiter=' ')
data2 = np.loadtxt('../shapelet_generation/shapelets_label-30.txt', dtype=int, delimiter=' ')
seq=5
batch=len(data1)
sequence_data=np.reshape(data1,(batch,seq,1))
sequence_label=data2

#################################################
'''
model = Sequential()
#model.add(Embedding(2000, 768, input_length=10))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
'''
model= load_model("my_mode.h5",custom_objects={'sparse_categorical_recall': km.sparse_categorical_recall,\
                                               'sparse_categorical_precision':km.sparse_categorical_precision, \
                                               'sparse_categorical_f1_score': km.sparse_categorical_f1_score, \
                                               'PositionalEmbedding':PositionalEmbedding })

#
# input_shape = sequence_data.shape[1:]
# def build_lstm( input_shape) :
#   inputs = keras.Input(shape=input_shape)
#   x = inputs
#   x=Bidirectional(LSTM(64,return_sequences=True))(x)
#   x=Dropout(0.5)(x)
#   x = Attention(128)(x)
#   x=Dropout(0.5)(x)
#   outputs=Dense(2, activation="softmax")(x)
#   for dim in [128]:
#       x = layers.Dense(dim, activation="relu")(x)
#       x = layers.Dropout(0.1)(x)
#   return keras.Model(inputs, outputs)
#
# model=build_lstm( input_shape)
#
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["sparse_categorical_accuracy",km.sparse_categorical_recall(),km.sparse_categorical_precision(),km.sparse_categorical_f1_score()],
)
# #model.summary()
#
# #callbacks = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
# #callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
#
# #checkpoint = ModelCheckpoint('{epoch}_model.h5',monitor='val_sparse_categorical_accuracy',save_best_only=True, period=1)



model = tensorflow.keras.models.Model(inputs=model.input, outputs=model.get_layer('tf.__operators__.add_3').output)


k=model.predict(
    data1,
    # sequence_label,
    # validation_split=0.4,
    # epochs=500,
    batch_size=128,
    # shuffle=True,
   # callbacks=[checkpoint],
)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
k=normalization(k)

m=2
k=np.reshape(k,(batch,seq))
data1 = pd.DataFrame(k)
data1.to_csv('unit_dim-30.csv')


#
# count=0
# for i in range (len(k)) :
#     if k[i][0]<0.5:
#         count=count+1
#
# print(len(k))
# print(count)

# model.save('my_mode.h5')

