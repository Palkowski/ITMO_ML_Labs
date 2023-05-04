import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imutils import paths


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


imagePaths = sorted(list(paths.list_images('train')))

X = []
y = []
for i in imagePaths:
    X.append(extract_histogram(cv2.imread(i)))
    if i.split(sep='.')[0].split(sep='\\')[1] == 'cat':
        y.append(0)
    else:
        y.append(1)

X = np.array(X)
y = np.array(y)
# print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=19)

clf = LinearSVC(C=0.6, random_state=19)
clf.fit(X_train, y_train)
thetas = clf.coef_.reshape(-1, 1)  # the hyperplane coefficients starting from theta_1
# print('thetas:', thetas, thetas.shape)
# theta_0 = clf.intercept_
# print('theta_0:', theta_0, theta_0.shape)
print('theta_11:', round(thetas[10][0], 2))
print('theta_13:', round(thetas[12][0], 2))
print('theta_318:', round(thetas[317][0], 2))

y_pred = clf.predict(X_test)
history = classification_report(y_test, y_pred, target_names=['cat', 'dog'])
print('\nlook at macro F1 here (f1-score column, macro avg row):\n', history)

imgs_to_check = ['test\\dog.1035.jpg', 'test\\dog.1022.jpg', 'test\\cat.1018.jpg', 'test\\cat.1002.jpg']
for i in imgs_to_check:
    print(i, 'class:', clf.predict([extract_histogram(cv2.imread(i))])[0])
