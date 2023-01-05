import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt

letters = pd.read_csv('letters.csv')
words = pd.read_csv('words.csv')
print(letters)
print(words)

clf = CategoricalNB(alpha=1, force_alpha=True)

