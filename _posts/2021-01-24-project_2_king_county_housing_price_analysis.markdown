---
layout: post
title:      "Project 2: King County Housing Price Analysis"
date:       2021-01-24 17:15:30 +0000
permalink:  project_2_king_county_housing_price_analysis
---


## Introduction
This project involves an iterative approach to building a multiple linear regression model to predict sale prices for houses in King County, WA by utilizing data of houses sold in 2014 and 2015. The dataset also includes over 21,000 entries with 21 variables/columns such as number of bedrooms, number of times the house was viewed before being bought, square footage of the home, the zip code that the house is located in, and many more. Throughout this project, I followed the OSEMN framework to find out which independent variables have the most influence in determining the sale price of a house.

## OSEMN Framework
![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/OSEMN.PNG)
### 1. Obtain
The very first step is straightforward. We obtain the data that we need from available data sources. In this case, King County House Prices dataset is provided in a CSV file to us by Flatiron School (and a similar dataset is available on Kaggle, in case you want to give it a try!)
```
df = pd.read_csv("kc_house_data.csv", index_col = "id") 
```

### 2. Scrub & Explore
These two steps tend to blend together as we focus on preprocessing our data and getting to know the dataset we are working with.

#### 2.1 - Null Values

I begin my data cleaning by calling the `.info()`command so I can confirm that each column has been entered in a way that makes sense respective to what the column measures. 

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/original%20dataset%20info.PNG)

Once I know my columns, I run `df.isnull().sum()` to find how many null values are in each column.

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/null%20sum.PNG)

Because there are about 21,000 total values in the dataset, we cannot drop all the rows containing null data as we would be sacrificing a huge portion of our dataset. I take a quick peek at the summary statistics for each of the columns containing null data (view, yr_renovated, and waterfront) and discovered that for all of the three variables in question, the overwhelming majority of the values are 0. Therefore, I can replace all of the null values with 0

#### 2.2 - Data Types

Since I now have a good idea of what information/measurements each column provides, I change the data types so there would be no ambiguity or error can occur when we plot our data later on.

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/data%20types.PNG)

#### 2.3 - Multicollinearity
Next, I check for multicollinearity between the columns. I create a table displaying which columns have correlation values higher than .75 with each other as well as a heatmap using seaborn to have a clear visual of which variables were redundant in my dataset. Then, I can drop the columns that would potentially cause my dataset to be unstable.

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/heatmap.png)

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/after%20heatmap.PNG)

#### 2.4 - Outliers
To discover if our dataset has outliers, I use `df.describe()` to look at the overall statistics information of each column.

I notice there’s a property that has 33 bedrooms, others have 11 or 10 bedrooms. Thesse could be an error when entering data (or it really is a huge mansion). So I decided to limit our dataset to 7 bedrooms. 
```
df = df[(df.bedrooms <= 7)]
```

I then plot a boxplot and distribution for price and look like we have a lot of high-value houses in our dataset which makes our distribution right-skewed. 

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/price%20distribution%20-%20right%20skewed.png)

This could ruin our results so I want to limit the sale price to $1,000,000. Now our price distribution is fairly normal and we can move on.

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/price%20distribution%20-%20normal.png)

#### 2.5 - Explore
In this step, I did a regression plot of each independent variable to the dependent variable (price) to discover any patterns and relationships. I, then, question which variables are categorical and which variables are numerical. 
For my categorical data, my data cleaning process involves binning about half of my categorical variables, converting the remaining integer data types to strings, and finally convert to dummy variables. I run`df.info()` and this is what I get.

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/categorical%20vs.%20numerical.PNG)

### 3. Model
This step consists of building and tuning models using all the tools you have in your data science toolbox. In this case, I drop any insignificant variables that have p-values greater than 0.05. I also drop any variables that could affect the overall model (such as condition variable because it can be a subjective opinion).

After comparing and testing different regression with different features, the final model has
* R-squared:  0.577
* Training Score:  0.5774
* Test Score: 0.5798
* P-values: all are less than 0.05

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/model.PNG)

#### 3.1 - Linear Regression Assumptions
We also want to make sure that our model met all 4 of linear regression assumptions
* Linear relationship: We check this assumption by exploring the regression plots between each independent variable vs. dependent variable (price)

* Independence: This is satisfied when we check the heatmap

![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/heatmap%20model.png)

* Homoscedasticity: This is checked by the residuals vs. fitted values plot. We can also use significance test like Goldfeld-Quandt to detect homoscedasticity. Since this test gives us a p-value > 0.05, the null hypothesis cannot be rejected, and we can assume the data is homoscedastic.
*
![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/Homoscedasticity.png)

* Normality: We use a Normal Q-Q plot and check to see the overall shape of our data against the required distribution to test the normality assumption. Our distribution is fairly normal and we can see how the quantiles of normal data appear as a straight line along the diagonal when plotted against a standard normal distribution's quantiles.
*
![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/QQ%20plot.png)

### 4. Interpret
* Multiple regression analysis was used to test if certain variables significantly predicted the sale price of homes in King County, Washington. The results of the regression indicated that 12 predictors can be explained 58% of the variance. All of the independent variables used in the model were significant predictors of sale price with p-values less than 0.05. All 4 assumptions of linear regression are met
* Looking at coefficients and predictors, we can conclude from our model that:
     * Houses that are renovated or built in the last 10 years increase the sale price of a home by 6682.52 dollars
     * One unit increase in square footage of internal living space increases the sale price by 149.64 dollars.
     
     ![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/living%20area.png)
		 
     * One additional bathroom increases the sale price by 7446.16 dollars.
     
     ![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/bathrooms.png)
		 
     * Houses that have longitude and latitude coordinates closer to big cities (Seattle or Bellevue) have higher prices
     
     ![](https://raw.githubusercontent.com/helenpham0229/Project2KingCountyImages/main/location.png)

## Future Work
* More research on the local economic indicators and housing market
* Do more research on school district as it could be an important variable in choosing a house
* Build a regression model to predict prices of homes over 1 million for high income families










