import numpy as np
from sklearn.metrics import confusion_matrix
import pylab as pl


def calc_confusion_matrix(labels, predcs):
    return confusion_matrix(y_true=labels, y_pred=predcs)


def plot_confusion_matrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix of the SID Classifier')
    pl.colorbar()
    pl.show()

# 测试数据
y_test=['business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business', 'business']

pred=np.array(['health', 'business', 'business', 'business', 'business',
       'business', 'health', 'health', 'business', 'business', 'business',
       'business', 'business', 'business', 'business', 'business',
       'health', 'health', 'business', 'health'])

cm = calc_confusion_matrix(y_test, pred)
plot_confusion_matrix(cm)