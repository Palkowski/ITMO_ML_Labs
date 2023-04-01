from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

c_data = pd.read_csv('candy-data.csv')
train_data = c_data[(c_data.competitorname != 'Chewey Lemonhead Fruit Mix') &
                    (c_data.competitorname != 'Runts') &
                    (c_data.competitorname != 'Sixlets')]
train_predictor = np.array(train_data.drop(columns=['competitorname', 'winpercent', 'Y']))
train_response = np.array(train_data.Y)

c_test = pd.read_csv('candy-test.csv')
test_predictor = np.array(c_test.drop(columns=['competitorname', 'Y']))
test_response = np.array(c_test.Y)

reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(train_predictor, train_response)

probs = reg.predict_proba(test_predictor)
comp_names = c_test.competitorname
thresholds = []
print('predicted [P(Y==0), P(Y==1)] for test data:')
for i in range(0, len(probs)):
    thresholds.append(probs[i, 1])
    print(comp_names[i], probs[i])
thresholds = sorted(thresholds)

print('\n')


def roc_curve(threshold, prnt=False):
    predicted_response = []
    for j in probs:
        if j[1] >= threshold:
            predicted_response.append(1)
        else:
            predicted_response.append(0)
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for j in range(0, len(test_response)):
        if predicted_response[j] == test_response[j] == 1:
            tp += 1
        elif predicted_response[j] == test_response[j] == 0:
            tn += 1
        elif test_response[j] == 1 and predicted_response[j] != test_response[j]:
            fn += 1
        elif test_response[j] == 0 and predicted_response[j] != test_response[j]:
            fp += 1
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    if prnt:
        print('\nthreshold:', threshold)
        print('TP, TN, FN, FP:', tp, tn, fn, fp)
        print('TPR:', tpr)
        print('Precision:', tp / (tp + fp))
    return [fpr, tpr]


roc_curve(0.5, True)

fpr_tpr = []
for i in thresholds:
    fpr_tpr.append(roc_curve(i))
fpr_tpr = np.array(fpr_tpr)
fpr_tpr.sort(axis=0)

fprs = fpr_tpr.T[0]
tprs = fpr_tpr.T[1]


def auc(x, y):
    s = 0
    for j in range(1, len(x)):
        s += y[j] * abs(x[j] - x[j - 1])
    return s


print('\nAUC:', auc(fprs, tprs))

plt.grid()
plt.plot(fprs, tprs)
plt.show()
