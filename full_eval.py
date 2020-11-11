
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np

def get_statistic(pred,target):
    score_dict={}
    n_class = target.shape[1]
    for i in range(n_class):
        acc=accuracy_score(target[:,i],pred[:,i])
        precision = precision_score(target[:, i], pred[:, i])
        recall = recall_score(target[:, i], pred[:, i])
        f1_value = f1_score(target[:, i], pred[:, i])
        # label = lb.classes_[i]
        score_dict[i] = (acc,precision, recall, f1_value)
    #print(score_dict)
    # 计算平均的指标
    acc=np.mean([item[0] for item in score_dict.values()])
    prec=np.mean([item[1] for item in score_dict.values()])
    recall=np.mean([item[2] for item in score_dict.values()])
    f1=np.mean([item[3] for item in score_dict.values()])
    return acc,prec,recall,f1

