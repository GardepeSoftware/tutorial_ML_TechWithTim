import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read in data from csv
data = pd.read_csv("./data/student-mat.csv", sep=";")
# Extract the attributes we're interested in
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# The value which we're trying to predict
predictAtt = "G3"

# We create an array of everything except the attribute we want to predict so
# we can use all the other attributes to predict its value
x = np.array(data.drop([predictAtt], 1))

# An array to hold values of attribute we want to predict
y = np.array(data[predictAtt])

# x_train == section of x data array
# y_train == section of y data array
# x_test and y_test are same but for running test after training
# We are reserving 10% of our data for testing (test_size)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # fit trains the model
    linear.fit(x_train, y_train)

    # score tests the model
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc

        # Saving model / "pickling it"
        # wb mode: creates file if it doesn't already exist?
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

# Open pickle file and load it
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


# Makes prediction of what the y value will be based on the inputs
predictions = linear.predict(x_test)

# Prints out the prediction, followed by the inputs, followed by the actual y value
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()



