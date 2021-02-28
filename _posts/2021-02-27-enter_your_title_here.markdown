---
layout: post
title:      "Project 3: Customer Churn Prediction"
date:       2021-02-27 20:31:26 -0500
permalink:  project_3_customer_churn_prediction
---


## Introduction
SyriaTel is a telecommunications company looking to predict and prevent customer churn. Customer churn is the percentage of customers that stopped using a company's product or service during a certain time frame. It can be a major problem because it impacts a company's customer loyalty and eventually affects company's revenue. 

To help SyriaTel fix the problem of customer churn, I did an Exploratory Data Analysis and then built a machine learning classifier that will predict whether the customers are going to churn. 

## Obtain
This project used the churn in telecoms dataset, which can be found in this repo (customer_churn_data.csv), and on kaggle via this [link](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset/code). This dataset included 21 columns and 3333 unique values. It was already clean with no outliers or null values.

## Scrub & Explore
In this portion, We learned that the churn percentage is about 14% or 483 out of 3333 customers. And we are going to explore different factors that can potentially lead to the mentioned rate. Here, I wonly highlight some interesting findings. You can find more details in the notebook.

### Account Length
We can see that account_length doesn't seem to have a significant effect whether a customer leaves the company or not because we have similar mean values and standard deviations for account length.

* Mean Account Length for Not Churn      : 3.36
* Std Dev Account Length for Not Churn : 1.33
* Mean Account Length for Churn              : 3.42
* Std Dev Account Length for Churn         : 1.32

![Alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/account%20length.png)

### State
It is clear that there are certain states with much higher churn. When grouped by state, CA, MD, NJ, TX have the highest churn percentages (approx 25%). States with the least churn include Alaska, Hawaii (approx 5%). However, we don't have information on "why" - it could be due to cell signal, different offers, etc.

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/state.png)

### International Plan
Out of 3333 customers, there are 323 people with international plan (42% of these customer churned) and 3010 people without this plan (11% of this group is no longer using SyriaTel's service). So, the percentage of customers who churn is higher for customers with international plans than for customers without international plans.

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/international%20plan.png)

### Voicemail Plan
Out of 3333 customers we observed, there are 922 customers with voicemail plan and 2411 customers without voicemail plan

The percentage of customers who churned is higher for customers without voicemail plans (17%)  than for customers with voicemail plans (9%)

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/voicemail%20plan.png)

People who receive more voicemail messages in average tend to churn more often.

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/voicemail%20messages.png)

### Customer Service Calls
When we look at customer service calls, we can see that as the number of customer service calls increases, the likelihood of churning increases as well. The majority of customers who DID NOT churn made 1-2 customer service calls. However, the majority people who DID churn made over 3 calls to customer service. 

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/customer%20calls.png)

### Total Calls
Customers who churned and those that did not churn had almost exactly the same usage across day, eve, night and international calls. 

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/total%20calls.png)

### Rates
* Day Rate: 0.17
* Eve Rate: 0.085
* Night Rate: 0.045
* International Plan Rate: 0.27
* Non-International Plan Rate:  0.27

The rates for international minutes are the same regardless of whether the customer has an international plan or not (27 cents per minute). 

## Model
In this portion of the project, I explored different classification models including logistic regression, KNN, random forest, and XGBoost. For each model, I observed the confusion matrix, ACU, precision, recall, accuracy, and F1 score for both training and testing splits.
* Confusion matrix: a table used to describe the performance of a classification model where one axis of the confusion matrix represents the ground-truth value of the items the model made predictions on, while the other axis represents the labels predicted by the classifier

* ROC/ ACU: the Receiver Operating Characteristic curve (ROC) which graphs the False Positive Rate against the True Positive Rate. The overall accuracy of a classifier can thus be quantified by the AUC, the Area Under the Curve. Perfect classifiers would have an AUC score of 1.0 while an AUC of 0.5 is deemed trivial or worthless. 

* Precision: measures how precise the predictions are (Number of true positives/Number of predicted positives)
* Recall:  indicates what percentage of the classes we're interested in were actually captured by the model (Number of true positives/Number of actual total positives)

* Accuracy: allows us to measure the total number of predictions a model gets right, including both True Positives and True Negatives (True positives + true negatives/total observations)

* F-1 Score: represents the Harmonic Mean of Precision and Recall. In short, this means that the F1 score cannot be high without both precision and recall also being high. When a model's F1 score is high, you know that your model is doing well all around.

## Interpretation
The best model was XGBoost with GridSearch tuning (with 0.93 ACU, 0.96 accuracy and 0.84 F1 score). 

![alt_text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/xgb%20gs%20confusion%20matrix.png)

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/xgb%20gs%20auc.png)

The top 3 important features are customer service calls, voicemail plan, and international plan.

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-3-Customer-Churn-Prediction/main/Images/feature%20importance.png)

## Recommendations
* Revisit and revise company’s customer service protocol (Possibly offer a larger incentive to customers making more than 3 customer service calls) 
* Changing the rates for international minutes/international plan because people with international plan pay at the same rate as people who don't have it
* Initiating customer feedback surveys for customers leaving



## Future Works


* Get more data on cell signal across the US to look for patterns in states with higher churn
* Information from customer service calls. I would like to know what the customers usually call about (Payments, Complaints, Web inquiries, etc.) and how the representatives handle each type of calls.
