import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# This basically just plotting the trend line and predict future input based on that trend line

# Reading input from the text file
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# Generating a function based on the input
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# plotting the results
plt.scatter(x_values, y_values)  # Make a scatter plot of the original data set
plt.plot(x_values, body_reg.predict(x_values))  # Plot the trendline
plt.show()  # Show the plot
