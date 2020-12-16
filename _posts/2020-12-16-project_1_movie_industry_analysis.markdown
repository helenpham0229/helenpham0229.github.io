---
layout: post
title:      "Project 1: Movie Industry Analysis"
date:       2020-12-16 15:43:09 +0000
permalink:  project_1_movie_industry_analysis
---


**Introduction**

As the final part of Module 1 in Flatiron School’s Data Science curriculum, I was assigned the project below:

“Microsoft sees all the big companies creating original video content, and they want to get in on the fun. They have decided to create a new movie studio, but the problem is they don’t know anything about creating movies. They have hired you to help them better understand the movie industry. Your team is charged with doing data analysis and creating a presentation that explores what type of films are currently doing the best at the box office. You must then translate those findings into actionable insights that the CEO can use when deciding what type of films they should be creating.”


**Technical Focus**

The focus of this project is EDA in Python. Therefore, I used the following libraries:
* Numpy: to perform a number of mathematical operations on arrays such as statistical and algebraic routines.
* Matplotlib: to visualize data in the form of graphs such as boxplot, barchart, line plot, etc.
* Pandas: it allows to manipulate and analyze data especially dataframes.
* Seaborn: simply put, to make our visualizations look more aesthetically pleasing.

**Exploring Data**

After importing all necessary libraries above, I imported the given dataset which are included a wide range of movie-related data in .csv files from Box Office Mojo, IMDB, Rotten Tomatoes, and TheMovieDB.org

Then, I used the .head() and .info() methods on all files to observe their columns, elements, and datatypes in order to know which files/data i can utilize for my analysis and whether I needed to collect new datasets.

For the files that I used, I changed data types on some, joined some dataframes, removed some columns, and tweaked some data to get the information needed for my analysis.

**Questions**

There are hundreds of factors that can contribute to the success of a movie. The classical factors include producer, production house, director, cast, runtime of the movie, the genre, the script, time of release.

In addition to the classical factors, there are a lot of social ones too. To name a few - the ratings, the viewer and critic reviews, the ongoing social, cultural, political and economic trends also are major deciding factors in the success of a film.

After exploring the given dataset, I decided to focus the following questions:

1. Which genres make the most profit?
2. When should we release our films?
3. Which director(s) should we hire?
4. Which studio are some of the biggest competitors?

**Results**

What are the most profitable genres?
![](https://drive.google.com/file/d/1Q-t5p0GO_zCcwbK0J2-8OKNGzwZ-1NLt/view?usp=sharing)
* After cleaning the dataset, we ended up with approximately 3500 movies
* The top 200 most profitable movies: Adventure: 117, Action: 87, Comedy: 50, Animation: 43, Sci-fi: 40
* Observing the bar chart, from the top 200 most profitable movies, Microsoft should invest in Adventure, Action, Sci-Fi, Animation, and Comedy movies

When should we release our productions?
![](https://drive.google.com/file/d/1ev5iBwyTnaPNQeyO_wzmE6Lu2nP7z7Yq/view?usp=sharing)
* I analyzed nearly 6000 films and their release dates
* June: $54M , July: $53M, November: $52M
* Based on the line plot, June, July, and November are the most profitable months to release movies.

Which director(s) should we hire?
* I observed around 3000 directors and their productions
* Directors who directed at least 4 movies tend to make positive net profit overall
![](https://drive.google.com/file/d/1BX1P9onA1jYqVDckSnPnUMDRdXEq8I98/view?usp=sharing)
* Out of those directors who directed 4 or more movies, these director make the most profit: Joss Whedon: $874M per production, Christopher Nolan: $864M per production, Jon Favreau: $826M per production
![](https://drive.google.com/file/d/1a7X4veMYGCn4cXH-wgCz0ffdO5mj2C5y/view?usp=sharing)

Who are our biggest competitors?
* I analyzed over 3000 movies from different studios
* From the top 50 most profitable movies: Buena Vista: 22, Warner Bros: 8, Universal: 5
![](https://drive.google.com/file/d/1mkZvK-giHdr6tBSziNIyXVPjZKk6Iead/view?usp=sharing)

**Future Work**

This analysis is just the beginning. In the future, I would want to 
* Collect more recent data from movies released in the past 3 years to keep up with the new trends/technology.
* Analyze budget details to see estimates in different stages of production
* Analyze the relationships or correlations between movies released only in theaters and movies released in both theaters and on streaming services (such as Netflix, Hulu, etc.)












