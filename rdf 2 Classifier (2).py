# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, classification_report

#Reading my data
df = pd.read_excel(r"D:\Gradution Project\EGX30.xlsx")
df.head()
#plot line graph of my data
df.plot.line(y="INDEXCLOSE",x="INDEXDATE")
plt.show()

#Change in price row
# calculate the change in price
df['CIP'] = df['INDEXCLOSE'].diff()

#Indicators Calculations 
#RSI : Relative Strength Index 

# Calculate the 14 day RSI
n = 14

# First make a copy of the data frame twice
up_df, down_df = df[['CIP']].copy(), df[['CIP']].copy()

# For up days, if the change is less than 0 set to 0.
up_df.loc['CIP'] = up_df.loc[(up_df['CIP'] < 0), 'CIP'] = 0

# For down days, if the change is greater than 0 set to 0.
down_df.loc['CIP'] = down_df.loc[(down_df['CIP'] > 0), 'CIP'] = 0

# We need change in price to be absolute.
down_df['CIP'] = down_df['CIP'].abs()

# Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
ewma_up = up_df['CIP'].transform(lambda x: x.ewm(span = n).mean())
ewma_down = down_df['CIP'].transform(lambda x: x.ewm(span = n).mean())

# Calculate the Relative Strength
relative_strength = ewma_up / ewma_down

# Calculate the Relative Strength Index
relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

# Add the info to the data frame.
df['down_days'] = down_df['CIP']

df['up_days'] = up_df['CIP']
df['RSI'] = relative_strength_index

# Display the head.
df.head(30)
#########################################
#Stochastic Oscillator 
# Calculate the Stochastic Oscillator
n = 14

# Make a copy of the high and low column.
low_14, high_14 = df[['INDEXLOW']].copy(), df[['INDEXHIGH']].copy()

# Group by symbol, then apply the rolling function and grab the Min and Max.
low_14 = low_14['INDEXLOW'].transform(lambda x: x.rolling(window = n).min())
high_14 = high_14['INDEXHIGH'].transform(lambda x: x.rolling(window = n).max())

# Calculate the Stochastic Oscillator.
k_percent = 100 * ((df['INDEXCLOSE'] - low_14) / (high_14 - low_14))

# Add the info to the data frame.
df['low_14'] = low_14
df['high_14'] = high_14
df['k_percent'] = k_percent

# Display the head.
df.head(30)
################################################################
#Williams %R
n = 14

# Make a copy of the high and low column.
low_14, high_14 = df[['INDEXLOW']].copy(), df[['INDEXHIGH']].copy()

# Group by symbol, then apply the rolling function and grab the Min and Max.
low_14 = low_14['INDEXLOW'].transform(lambda x: x.rolling(window = n).min())
high_14 = high_14['INDEXHIGH'].transform(lambda x: x.rolling(window = n).max())

# Calculate William %R indicator.
r_percent = ((high_14 - df['INDEXCLOSE']) / (high_14 - low_14)) * - 100

# Add the info to the data frame.
df['r_percent'] = r_percent

# Display the head.
df.head(30)
#############################################################
#Moving Average Convergence Divergnece (MACD)
# Calculate the MACD
ema_26 = df['INDEXCLOSE'].transform(lambda x: x.ewm(span = 26).mean())
ema_12 = df['INDEXCLOSE'].transform(lambda x: x.ewm(span = 12).mean())
macd = ema_12 - ema_26

# Calculate the EMA
ema_9_macd = macd.ewm(span = 9).mean()

# Store the data in the data frame.
df['MACD'] = macd
df['MACD_EMA'] = ema_9_macd

# Print the head.
df.head(30)
############################################
# Price Rate Of Change
# Calculate the Price Rate of Change
n = 9

# Calculate the Rate of Change in the Price, and store it in the Data Frame.
df['Price_Rate_Of_Change'] = df['INDEXCLOSE'].transform(lambda x: x.pct_change(periods = n))

# Print the first 30 rows
df.head(30)
########################################
#BUILDING THE MODEL
# CLASSIFICATION PROBLEM
# Create a column we wish to predict
'''
    In this case, let's create an output column that will be 1 if the closing price at time 't' is greater than 't-1' and 0 otherwise.
    In other words, if the today's closing price is greater than yesterday's closing price it would be 1.
'''

# Group by the `Symbol` column, then grab the `Close` column.
close_groups = df['INDEXCLOSE']

# Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

# add the data to the main dataframe.
df['Prediction'] = close_groups

# for simplicity in later sections I'm going to make a change to our prediction column. To keep this as a binary classifier I'll change flat days and consider them up days.
df.loc[df['Prediction'] == 0.0] = 1.0

# print the head
df.head(50)

# OPTIONAL CODE: Dump the data frame to a CSV file to examine the data yourself.
#df.to_csv('final_metrics.csv')

#REMOVING NANS
# We need to remove all rows that have an NaN value.
print('Before NaN Drop we have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

# Any row that has a `NaN` value will be dropped.
df = df.dropna()

# Display how much we have left now.
print('After NaN Drop we have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

# Print the head.
df.head()

#Building the model

df['INDEXDATE'].describe()
train_data = df[(df['INDEXDATE'] >= '2016-01-01') & (df['INDEXDATE'] <= '2021-12-26')]
test_data = df[(df['INDEXDATE'] >= '2022-01-01') & (df['INDEXDATE'] <= '2024-02-28')]

X_train = train_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD']]
Y_train = train_data['Prediction']

X_test = test_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD']]
Y_test = test_data['Prediction']

# Grab our X & Y Columns.
#X_Cols = df[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD']]
#Y_Cols = df['Prediction']

# Split X and y into X_
#X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

# Create a Random Forest Classifier
rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

# Fit the data to the model
rand_frst_clf.fit(X_train, Y_train)

# Make predictions
y_pred = rand_frst_clf.predict(X_test)

# Print the Accuracy of our Model.
print('Correct Prediction (%): ', accuracy_score(Y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0)


###################################################################################################################################
# Define the traget names
target_names = ['Down Day', 'Up Day']

# Build a classifcation report
report = classification_report(y_true = Y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

# Add it to a data frame, transpose it for readability.
report_df = pd.DataFrame(report).transpose()
report_df

#Model Evaulation: Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

rf_matrix = confusion_matrix(Y_test, y_pred)

true_negatives = rf_matrix[0][0]
false_negatives = rf_matrix[1][0]
true_positives = rf_matrix[1][1]
false_positives = rf_matrix[0][1]

accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
percision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)

print('Accuracy: {}'.format(float(accuracy)))
print('Percision: {}'.format(float(percision)))
print('Recall: {}'.format(float(recall)))
print('Specificity: {}'.format(float(specificity)))

#Feauture Importance 
# Calculate feature importance and store in pandas series
feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_imp
# store the values in a list to plot.
x_values = list(range(len(rand_frst_clf.feature_importances_)))

# Cumulative importances
cumulative_importances = np.cumsum(feature_imp.values)

# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')

# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')

# Format x ticks and labels
plt.xticks(x_values, feature_imp.index, rotation = 'vertical')

# Axis labels and title
plt.xlabel('Variable')
plt.ylabel('Cumulative Importance')
plt.title('Random Forest: Feature Importance Graph')

plt.show()
pip install scikit-plot

# Create an ROC Curve plot.
import scikitplot as skplt


from scikitplot.roc import plot_roc_curve

import matplotlib.pyplot as plt

rfc_disp = plot_roc_curve(rand_frst_clf, X_test, Y_test, alpha = 0.8)
plt.show()

#Model Improvement 
# Number of trees in random forest
# Number of trees is not a parameter that should be tuned, but just set large enough usually. There is no risk of overfitting in random forest with growing number of # trees, as they are trained independently from each other. 
n_estimators = list(range(100, 2000, 200))

# Number of features to consider at every split
max_features = ['auto', 'sqrt', None, 'log2']

# Maximum number of levels in tree
# Max depth is a parameter that most of the times should be set as high as possible, but possibly better performance can be achieved by setting it lower.
max_depth = list(range(10, 110, 10))
max_depth.append(None)

# Minimum number of samples required to split a node
# Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can also lead to # under-fitting hence depending on the level of underfitting or overfitting, you can tune the values for min_samples_split.
min_samples_split = [2, 5, 10, 20, 30, 40]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 7, 12, 14, 16 ,20]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

# New Random Forest Classifier to house optimal parameters
rf = RandomForestClassifier()

# Specfiy the details of our Randomized Search
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, Y_train)

# With the new Random Classifier trained we can proceed to our regular steps, prediction.
rf_random.predict(X_test)

# Print best parameters found by RandomizedSearchCV
print("Best Parameters:", rf_random.best_params_)



'''
    ACCURACY
'''
# Once the predictions have been made, then grab the accuracy score.
print('Correct Prediction (%): ', accuracy_score(Y_test, rf_random.predict(X_test), normalize = True) * 100.0)


'''
    CLASSIFICATION REPORT
'''
# Define the traget names
target_names = ['Down Day', 'Up Day']

# Build a classifcation report
report = classification_report(y_true = Y_test, y_pred = y_pred, target_names = target_names, output_dict = True)


# Add it to a data frame, transpose it for readability.
report_df = pd.DataFrame(report).transpose()
display(report_df)
print('\n')


'''
    FEATURE IMPORTANCE
'''
# Calculate feature importance and store in pandas series
feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
display(feature_imp)


'''
    ROC CURVE
'''

fig, ax = plt.subplots()

#MMKN 7ad Y RUN DA
# Create an ROC Curve plot.
rfc_disp = plot_roc_curve(rand_frst_clf, X_test, Y_test, alpha = 0.8, name='ROC Curve', lw=1, ax=ax)

# Add our Chance Line
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

# Make it look pretty.
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve Random Forest")

# Add the legend to the plot
ax.legend(loc="lower right")

plt.show()