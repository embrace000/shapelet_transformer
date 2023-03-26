import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import random
from pyts.transformation import ShapeletTransform
import numpy as np
import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.transformation import ShapeletTransform
from joblib import Parallel, delayed

if __name__ == '__main__':


 data1=np.loadtxt('../data_sequence/sequence1.txt',dtype=int,delimiter=' ')
 data1_spilt = np.array_split((data1), 20)
 y1=np.zeros(len(data1))
 y1_spilt=np.array_split((y1), 20)

 data2 =np.loadtxt('../data_sequence/sequence2.txt', dtype=int, delimiter=' ')
 data2_spilt = np.array_split((data2), 20)
 y2=np.ones(len(data2))
 y2_spilt = np.array_split((y2), 20)

 for k in range(20) :
  X = np.vstack((data1_spilt[k], data2_spilt[k]))
  y = np.hstack((y1_spilt[k], y2_spilt[k]))
  st = ShapeletTransform(n_shapelets=10, window_sizes=[5])
  st.fit_transform(X, y)
  data = pd.DataFrame(st.shapelets_).astype(int)
  data.to_csv('../shapelet_generation/shapelets_'+str(k)+'.txt', sep=' ', index=False, header=False)
  print(k)










 #
 # X=np.vstack((data1,data2))
 # y=np.hstack((y1,y2))
 #
 # st = ShapeletTransform(n_shapelets=20, window_sizes=[5])
 # st.fit_transform(X, y)
 #
 #
 # def fit_transform(X, y):
 #  return st.fit_transform(X, y)
 # #
 # # results = Parallel(n_jobs=-1)(delayed(fit_transform)(X, y) for _ in range(10))
 # #
 # results = Parallel(n_jobs=32)(delayed(fit_transform)(X[i::32], y[i::32]) for i in range(32))
 #
 # data = pd.DataFrame(st.shapelets_ ).astype(int)
 # data.to_csv( 'shapelet_generation/shapelets.txt', sep=' ', index=False, header=False)




 # data = np.loadtxt('shapelets.txt',dtype=int,delimiter=' ')

 # X = [[0, 200, 31, 4, 3, 2, 1],
 #      [0, 1, 3, 4, 3, 4, 5],
 #      [2, 1, 0, 200, 30, 4, 5],
 #      [1, 2, 2, 1, 0, 3, 5]]
 # y = [1, 0, 1, 0]
 # X=np.random.randint(0,10,size=(10,20))
 # y=np.random.randint(0, 2, size=10)
















#################################################################################
 # # Toy dataset
 # X_train, _, y_train, _ = load_gunpoint(return_X_y=True)
 #
 # # Shapelet transformation
 # st = ShapeletTransform(window_sizes=[12, 24, 36, 48],
 #                        random_state=42, sort=True)
 # X_new = st.fit_transform(X_train, y_train)
 #
 # # Visualize the four most discriminative shapelets
 # plt.figure(figsize=(6, 6))
 # for i, index in enumerate(st.indices_[:6]):
 #     idx, start, end = index
 #     plt.plot(X_train[idx], color='C{}'.format(i),
 #              label='Sample {}'.format(idx))
 #     plt.plot(np.arange(start, end), X_train[idx, start:end],
 #              lw=5, color='C{}'.format(i))
 #
 # plt.xlabel('Time', fontsize=12)
 # plt.title('The four most discriminative shapelets', fontsize=14)
 # plt.legend(loc='best', fontsize=8)
 # plt.show()
 # print('3')