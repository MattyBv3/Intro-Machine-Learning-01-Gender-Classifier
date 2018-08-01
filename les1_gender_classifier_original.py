#import tree sub-module from sklearn (machine learning models)
from sklearn import tree

#create data set - list of 9 lists (data type that stores sequence of values)
# [height in inches, weight in lbs, shoe size in inches]
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
[190, 90, 47], [175, 64, 39], [177, 70, 40], [171, 75, 42]]

# create list of 9 corresponding labels...gender in this case
y = ['male', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'female']

# create variable clf (classifier) to store decision tree model
# reference tree dependancy
# initialize decision tree by calling decision tree method on tree object
# variable = object.method
clf = tree.DecisionTreeClassifier()

# now have variable, can train it on data set
# call .fit method on clf variable, takes 2 arguments, result stored in updated clf variable
# .fit method trains decision tree on our data set
clf = clf.fit(x,y)

# data set is now trained using .fit method

# new var prediction, call .predict method on clf variable
# enter new data set
prediction = clf.predict([[190,70,43]])
print(prediction)
