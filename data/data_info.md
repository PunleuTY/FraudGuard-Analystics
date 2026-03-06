# Infomation of dataset

Based on the original dataset, we got two csv files such as fraudTest.csv and fraudTrain.csv. This format is typically the dataset that used in the Kaggle Competition.
The purpose of spliting dataset is because we would want to test our model with unseen data, to evaluate the performance of model and also detecting the overfitting case.

```Train dataset
Train Dataset (fraudTrain.csv), typically is the file that used build and train our ML model.
- It usually contains: Feature columns (input variables) and the target variable or column (i.g,. is_fraud which classifies into 0 or 1)
```

```Test dataset
Test Dataset (fraudTest.csv) is used to evaluate the trained model.
- It contains same feature (input variable) as the train set BUT there typically is not include the target variable.

BUT in our case as we've inspect the fraudTest.csv then we've seen the target variable. SO it is because this dataset was split which technically called Pre-Split dataset.

THEN we can moving into our defined step, but not all the next steps we have to working on only the train dataset and the test dataset will be used in our final evaluation (testing with unseen dataset after we train our model with the fraudTrain.csv) 

```
