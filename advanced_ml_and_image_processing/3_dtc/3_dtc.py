import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

# 1
data = pd.read_csv('diabetes.csv')
data_690 = data[data.index < 690]
print('class 0 of data_690:', data_690[data_690['Outcome'] == 0].shape[0])

# 2
data_690_train = data_690[data_690.index < (690 * 80 // 100)]
data_690_test = data_690[data_690.index >= (690 * 80 // 100)]
y_train = data_690_train['Outcome']
X_train = data_690_train.drop('Outcome', axis=1)
y_test = data_690_test['Outcome']
X_test = data_690_test.drop('Outcome', axis=1)
clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=20, min_samples_leaf=10, random_state=2020)
clf.fit(X_train, y_train)

columns = list(X_train.columns)
export_graphviz(clf, out_file='tree.dot',
                feature_names=columns,
                class_names=['0', '1'],
                rounded=True, proportion=False,
                precision=2, filled=True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

s = graphviz.Source(dot_graph)
s.view()
print('depth:', clf.get_depth())

# 3
# look at the graph
print('Bottom level predictor: Glucose')

# 4
# look at the graph
print('Bottom level predictor value:', 142.5)

# 5
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
# look at accuracy
print('Accuracy:', 0.79)

# 6
# look at "macro avg" row and "f1-score" column
print('Macro f-1:', .75)

# 7, 8, 9, 10
data_check = data.iloc[[727, 710, 704, 729]].drop('Outcome', axis=1)
print('Patients 727, 710, 704, 729:', clf.predict(data_check))
