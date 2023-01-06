import numpy as np
import pandas as pd

letters = pd.read_csv('letters.csv')
words = pd.read_csv('words.csv')
print(letters)
print(words)

# probability that the letter is spam
p_spam = letters.spam[0] / (letters.spam[0] + letters.not_spam[0])
p_not_spam = letters.not_spam[0] / (letters.spam[0] + letters.not_spam[0])


# probability of encountering the word "x" in spam letters with Laplace smoothing
def p_x_spam(x, r=0):
    row = words[words.word == x]
    value = np.array(row.spam)
    if len(value) == 0:
        value = 0
    else:
        value = value[0]
    return (value + 1) / (letters.spam[1] + r + words.word.shape[0])


# probability of encountering the word "x" in not_spam letters with Laplace smoothing
def p_x_not_spam(x, r=0):
    row = words[words.word == x]
    value = np.array(row.not_spam)
    if len(value) == 0:
        value = 0
    else:
        value = value[0]
    return (value + 1) / (letters.not_spam[1] + r + words.word.shape[0])


string_to_check = 'Remove Million Bonus Access Online Money Bill'
list_of_words = string_to_check.split(sep=' ')

not_in_list = 0
for i in list_of_words:
    if i not in list(words.word):
        not_in_list += 1
        print(i, 'not in list')

print('P(spam):', p_spam)

f_spam = np.log(p_spam)
for i in list_of_words:
    f_spam += np.log(p_x_spam(i, not_in_list))
print('F(spam):', f_spam)

f_not_spam = np.log(p_not_spam)
for i in list_of_words:
    f_not_spam += np.log(p_x_not_spam(i, not_in_list))
print('F(not_spam):', f_not_spam)

p_string_spam = np.exp(f_spam) / (np.exp(f_spam) + np.exp(f_not_spam))
print('P(\"Remove Million Bonus Access Online Money Bill\" = spam):', p_string_spam)
