
import numpy as np 
import pandas as pd 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Create train and test datasets
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(9)
train_data = train_data.copy()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

labels = ['Pclass', #Pclass
          'Sex', #Sex (changed to binary 0-1)
          'Age'] #Age
          
train_data.replace('male', 1, inplace = True) # As the model prefers numerical values, 'male'
train_data.replace('female', 0, inplace = True) #  and 'female' labels are changed to 1 and 0

# Same for test_data

test_data.replace('male', 1, inplace = True)
test_data.replace('female', 0, inplace = True)

X = train_data[labels]
y = train_data.Survived # Create copy for further changes 

# Remove missing values

# X.iloc[1]['Age'] # This is a way of selecting each value on a column

list_missing_values = [] #List with each index on which the Age is missing
for i in range(891):
    if X.isna().iloc[i]['Age'] == True:
        list_missing_values.append(i)

# Now we will delete these rows for y
# To do this, I will replace the values in y for each missing value index 
# with None, and then apply the dropna() function

for j in list_missing_values:
    y[j] = None
    
X = X.dropna()
y = y.dropna()

# Sklearn model

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver='sgd', verbose=True, random_state=1, tol=0.0002, max_iter= 500)
model.fit(X, y)

# Check results

test_X = test_data[labels]

mean = test_X['Age'].mean()
std = test_X['Age'].std()

# Function to fill the missing values with a Gaussian distribution
def fill_missing_from_Gaussian(column_val):
    if np.isnan(column_val) == True: 
        column_val = int(abs(np.random.normal(mean, std, 1))) # The model does not accept negative numbers
    else:
         column_val = column_val
    return column_val

test_X['Age'] = test_X['Age'].apply(fill_missing_from_Gaussian) 

test_predict = model.predict(test_X).astype(int)
print(test_data)
print(test_predict)

# Fill the csv file with the results
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predict})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# Check the format and length of the file is the correct one
submission = pd.read_csv('my_submission.csv')
submission.head()
submission
