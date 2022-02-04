# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Baseline Logistic Regression model from sklearn, [ref](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
<br>

`LogisticRegression(random_state = 0,solver = 'lbfgs', multi_class = 'auto')` 

## Intended Use
This model predicts a person's income range.

## Training Data
The census data is available at the UCI library [Link](https://archive.ics.uci.edu/ml/datasets/census+income).

## Evaluation Data
The evaluation data is splitted from the training set, use 80:20 ratio.

## Metrics
precision = 0.73,  recall = 0.27 and fbeta = 0.4

## Ethical Considerations
Public dataset, all the rights from the dataset is reserved to UCI and should be refenced.

## Caveats and Recommendations
Data needs to be clean before use. 
Skipp all white space and remove all missing values. 