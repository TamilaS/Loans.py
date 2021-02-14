import numpy as np
import pandas as pd
from sklearn import tree, ensemble

df=pd.read_excel('loandata.xlsx')
df.head(10)
df=pd.get_dummies(df)
df=df.drop(['Default_No'], axis=1)

Xy=np.array(df)
seed = np.random.seed(2)
np.random.shuffle(Xy)
X=Xy[:,:-1]
y=Xy[:,-1]

trainsize = 1000
trainplusvalsize = 500
X_train=X[:trainsize]
X_val=X[trainsize:trainsize + trainplusvalsize]
X_test=X[trainsize + trainplusvalsize:]

y_train=y[:trainsize]
y_val=y[trainsize:trainsize + trainplusvalsize]
y_test=y[trainsize + trainplusvalsize:]

acc_train = 1-sum(y_train)/len(y_train)
acc_val = 1-sum(y_val)/len(y_val)

print ( 'Naïve guess train and validation', acc_train , acc_val)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

print ( 'Full tree guess train/validation ',clf.score(X_train, y_train),clf.score(X_val, y_val))


bestdepth=-1
bestscore=0
max_depth = 15

for i in range(max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=i+1)
    clf.fit(X_train,y_train)
    trainscore=clf.score(X_train,y_train)
    valscore=clf.score(X_val,y_val)
    print( 'Depth:', i+1, 'Train Score:', trainscore, 'Validation Score:', valscore)

    if valscore>bestscore:
        bestscore=valscore
        bestdepth=i+1


X_trainval=X[:trainplusvalsize,:]
y_trainval=y[:trainplusvalsize]


clf = tree.DecisionTreeClassifier(max_depth=bestdepth)
clf.fit(X_trainval,y_trainval)


test_score = clf.score(X_test,y_test)
print('testing set score', test_score)
