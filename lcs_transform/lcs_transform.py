import numpy
import numpy as np
import pandas as pd
import lcs
from  tqdm import tqdm
if __name__ == '__main__':

  shapelets = np.loadtxt('shapelet_generation/shapelets.txt',dtype=int,delimiter=' ')
  data1=np.loadtxt('data_sequence/sequence4.txt',dtype=int,delimiter=' ')
  # data2 = np.loadtxt('sequence2.txt', dtype=int, delimiter=' ')
  # data3 = np.loadtxt('sequence3.txt', dtype=int, delimiter=' ')

  data1=data1.reshape(len(data1)*20)
  # data2=data2.reshape(len(data2)*20)

  temp1=np.array([])
  temp2 = np.array([])
  temp3 = np.array([])
  temp4= np.array([])
  temp5 = np.array([])

  for i in tqdm (range(int(int(len(data1)/40)))) :
      l1=[]
      l1 =data1 [i*40:(i+1)*40]

      for j in range(len(shapelets)):
         l2  =   shapelets[j]
         length_,lists = lcs.calculate_LCS(l1, l2)
         if len(lists)==5 :
            temp1 = np.append( temp1,j).astype(int)
            temp2 = np.append(temp2 ,np.where(l1 == shapelets[j][0])[0][0]).astype(int)
            # print(lists)

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
  seqlen = 10 #input_size
  temp6= np.array([])
  temp7 = np.ones(seqlen)

  def bound_diff(index):
      if index > 10:
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
  data.to_csv('lcs_transform/transform4.txt', sep=' ', index=False, header=False)



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
