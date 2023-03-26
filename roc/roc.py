# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


y_test = np.loadtxt('roc-label-wo-0.2.csv')
y_score= np.loadtxt('roc-score-wo-0.2.csv')


y_test2 = np.loadtxt('roc-label-wi-0.2.csv')
y_score2= np.loadtxt('roc-score-wi-0.2.csv')







# 计算
fpr, tpr, thread = roc_curve(y_test, y_score)

roc_auc= auc(fpr, tpr)

fpr2, tpr2, thread2 = roc_curve(y_test2, y_score2)

roc_auc2= auc(fpr2, tpr2)

# 绘图
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Without behavior unit feathers (area = %0.2f)' % roc_auc)
plt.plot(fpr2, tpr2, color='red',
         lw=lw, label='With behavior unit feathers (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.legend(loc="lower right")
plt.savefig('roc.png',)
plt.show()
