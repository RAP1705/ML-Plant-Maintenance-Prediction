# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import warnings
# import pickle
# warnings.filterwarnings("ignore")



# # Load the dataset using the pandas & convert into numpy array
# data = pd.read_csv("ML_Data.csv")
# data = np.array(data)


# # extract X (Probable Inputs) values and Y (Probable Outputs) and convert into Integers
# X = data[1:, 1:-1]
# y = data[1:, -1]
# y = y.astype('int')
# X = X.astype('int')

# print(X, y)


# # Split the dataset into the Train(70%) and Test(30%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# log_reg = LogisticRegression()

# # Fed the train dataset int othe Logistic Regression Algo from Sklearn libtrary
# log_reg.fit(X_train, y_train)

# # Use teh score method to get the accuracy of the model
# score = log_reg.score(X_test, y_test)
# print(score)


# # Test out with your sample data and get the probablity
# inputt=[int(x) for x in "200 70 800 5 10000 140".split(' ')]
# final=[np.array(inputt)]
# b = log_reg.predict_proba(final)
# output='{0:.{1}f}'.format(b[0][1], 2)
# print(output)


# # Generate custom ML model in pickle file
# pickle.dump(log_reg,open('model.pkl','wb'))

# # Example: how to read the custom model (pickle file) form your actual application
# model=pickle.load(open('model.pkl','rb'))




# ######------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset using pandas
data = pd.read_csv("ML_Data.csv")
data = np.array(data)

# Extract X (Features) and y (Labels)
X = data[1:, 1:-1]
y = data[1:, -1].astype('int')
X = X.astype('int')

# Print X and y to check data
print(X, y)

# Optional: Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into Train (70%) and Test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Evaluate the model
score = log_reg.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# Test with a sample input and get the probability
inputt = [int(x) for x in "200 70 800 5 10000 140".split(' ')]
final = [np.array(inputt)]
final_scaled = scaler.transform(final)  # Scale the input like the training data

# Predict probability
b = log_reg.predict_proba(final_scaled)
output = '{0:.2f}'.format(b[0][1])
print(f"Predicted Probability: {output}")

# Save the model to a pickle file
# pickle.dump(log_reg, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


# Example: How to load the model from a pickle file
model = pickle.load(open('model.pkl', 'rb'))
