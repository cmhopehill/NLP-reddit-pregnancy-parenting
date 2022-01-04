# Pregnancy and Parenting: A Natural Language Processing(NLP) Classification Project on r/pregnant and r/parenting

## Problem Statement
The goal of this project is to gather data from two different subreddits ([*r/pregnant*](https://www.reddit.com/r/Parenting/) & [*r/parenting*](https://www.reddit.com/r/pregnant/)) and then use Natural Language Processing (NLP) to train a classifier to be able to determine which subreddit a given post comes from. I chose r/parenting and r/pregnant as comparisons for this project as my partner and I are currently expecting our first child and I was interested in looking at the differences in language for individuals posting about their pregnancy on r/pregnant and those who posting on r/parenting. To examine the differences I will be using classification models (such as Random Forests and Logistic Regression) to train the model on the data from each subreddit and then attempt to accurately classify which sub a post or string is likely to be from. A successful project will be one where the classification model is able to accurately predict which subreddit a post comes from based on its text above baseline, but ideally with an 80% or better accuracy. I believe the results of this project may give first-time parents an idea of the changes that occur as individuals transition from pregnancy to becoming a parent. This project may also have benefits to parents who are considering having another child to remind them of the similarities and differences between being pregnant and being/becoming a parent.

## Background
The initial idea for the comparison of r/pregnant and r/parenting came out of the personal relevance of the topic to my current life situation. After thinking more about how individuals have or have not been able to find connection since the advent of the COVID-19 pandemic, it occured to me that Reddit and other social media outlets are the major outlets that most people have had to find community and share their current experiences. With that in mind a comparison of these subreddits may also have real meaning and impact for many people who are feeling cut off from typical sources of community for parents and parents to be.

### Subreddit Information
* r/pregnant:
	* Created on April 20, 2009
	* 99.5 thousand members
	* Headline: “A safe, welcoming community for all pregnant people!”
* r/parenting:
	* Created on March 24, 2008
	* 3.6 million members
	* Reddit Parenting - For those with kids of any age!

## Structure of Repo
The notebooks in this repo are numbered from 1-4 to indicate both the order in which to read them, they also indicate the order in which I went through the Data Science process. At different points in each notebook there may be code that has been commented out, when this occurs there will also be a comment explaining why the code was commented out and what it did. In addition to the numbered notebooks there is also a web scrape notebook that illustrates the data collection process I went through using the ([Pushshift API](https://github.com/pushshift/api)) to scrape data from the r/pregnant and r/parenting subreddits. Along with the notebooks you will also find a visuals folder, and a pdf of my GA class presentation for this project. The data folder for for all of the data used in this project can be found [here](https://drive.google.com/drive/folders/1jTaAQyCF7J2xsVsBQNUOKwfzB9Xv3Qtp?usp=sharing) and is broken down into three subfolders for the raw data, lemmatized or lem_data, and a downsampled folder with the downsampled datasets that I used for the modeling part of this project.

## Pushshift API and Data Collection
Using the Pushshift API and with assistance from some fellow GA classmates (Jeffrey Floyd and Mark Harris) I created a two step web scrape function that I used to pull 100,000 times from both r/pregnant and r/parenting. After examining the multiple columns that were pulled from an initial scrape I made the decision to only have the web scrape function pull the following:
* author (author of the post)
* subreddit (subreddit the posts were scraped from) 
    * will be used for data check and classification modeling
* title (title of the post)
    * use as additional text data if needed
* selftext (the text of the post)
    * main avenue of text data
* created_utc (unix time code or the time that the post was posted to reddit)
    * needed for the Pushshift function to ensure that each consecutive scrape would attempt to pull new data based on pulling from before the lowest utc from the previous loop.

## Data Cleaning and EDA
After  collecting the data using the Pushshift API and web scrape functions I then moved on to cleaning and checking the data that had been pulled from each subreddit. Initial data checks were done to look at each subreddit for total number of unique posts, time frame of posts, and the distribution of posts by author:
* r/pregnant
	* Time frame from May 27, 2017 to October 30, 2021
	* 94,671 unique posts out of 99,978 total posts
	* 35,415 unique redditors have posted on r/pregnant, but of those only the top 1500 have made ten or more posts.

* r/parenting
	* Time frame from February 10, 201 to October 30, 2021
	* 79,174 unique posts out of 99,988 total posts
	* 51,150 unique redditors have posted on r/pregnant, but of those only the top 385 have made ten or more posts.

Following this initial data investigation of each subreddit I then moved on to the data cleaning by checking for null values and any posts that had been removed. I then moved into lemmatizing the data to prepare for modeling, however I soon realized that I still had a number of posts that were causing issues as I kept getting long strings of numbers and letters in my list of lemmatized features. After consulting with one of my classmates we realized that a subset of the posts for each reddit may have had hyperlinks that when lemmatized were leading to the number string issues. I then dropped all of the posts that had hyperlinks in them, as they were causing issues as lems, but it also looked as though some of these posts were essentially ads or endorsements for different products. After this final cleaning I then used WordNet Lemmatizer and Regex Tokenizer to tokenize and lemmatize the title text and post text for both subreddits, and then exported those new dataframes as pregnant_lems and parent_lems for use in the next notebooks.

## Preprocessing
After reading in the new lemmatized data for both subreddits I then created two new columns ‘all_lems’ for the combination of the lems from each post and its title, and ‘word_count’ as a total number of words from all_lems. I then examined the word count distributions for both subreddits to look for outliers and check for how similar or different the posts were in word count for each subreddit. As it turns out the box plots for both r/pregnant and r/parenting had a similar level of spread and similar high and low outliers. Seeing this I then investigated further to find a word count high and low that could be used to get rid of outliers. I did this in an attempt to shrink the overall data size and make it more computationally manageable for my computer. After setting word count parameters of 40-400 for r/pregnant and 50-400 for r/parenting I still had 80,000+ posts and 65,000+ respectively. This still seemed like too much data to effectively and efficiently grid search and potentially model on so I made the decision to use pandas .sample function to randomly sample 25,000 posts from each subreddit that I then merged for a merged data frame of 50,000 posts from each subreddit. I then exported this lemmatized and downsampled data frame to a csv as preg_parent_downsample for use in the modeling notebooks.

## Modeling
After  finishing up the preprocessing of the data I then moved on to modeling with different classifiers. I broke the modeling into two notebooks with Logistic Regression and Naive Bayes being used as classifiers in notebook 3 and Random Forests and Extra Trees being used as classifiers in notebook 4. In both cases I used a train and test split of ⅔ : ⅓ and stratified y.

### Model 1 Logistic Regression and Count Word Vectorizer
The first model I attempted was a pipelined model using Logistic Regression and Count Word Vectorizer. This model performed very well with a training set score of .968 and a testing set score of .950. While the model is a tiny bit overfit, the difference is less than 2% and it is an overall very strong model.

### Model 2 Naive Bayes and Tfidf Vectorizer
The next model I looked at was a pipeline model using Naive Bayes and Tfidf Vectorizer. This model also performed very well overall, though slightly worse than model 2, with a train score of .931 and a test score of .917.

### Model 3 Logistic Regression and Countvectorizer
The final model in notebook 3 was a pipeline model using Logistic Regression again, but this time with Tfidf Vectorizer. This proved to be slightly better than model 1 and the strongest overall model with a train score of .972 and a test score of .952. This model is ever so slightly more overfit when compared to model 1, but both the training and test scores were higher so it's the model I would recommend in use when comparing r/pregnant and r/parenting.

### Random Forest and Extra Trees 
Notebook 4 examines two ensemble decision tree models with Random Forest Classifier and Extra Trees classifier. Ultimately neither of these models performed very well with both scoring out as heavily overfit to the training data. Both had training scores nearing 1, but test scores around .62-.63. I attempted to run some grid searches to try and attempt to improve these models, but they ultimately did not lead to much improvement when compared to those models in notebook3.

## Conclusions, Recommendations and Future Directions
Logistic Regression paired with Tfidf Vectorizer led to an incredibly powerful model that was able to predict at over 95% on either subreddit based on an F1 score of .952. Based on this I would recommend the use of these two methods together when attempting to predict if a particular post is from r/pregnant or r/parenting. This model performed immensely better than the baseline model and also went well past what I had termed as success in my initial problem statement.

If I had further time for this project I would consider three things: First using steamlit to deploy this model for use. Secondly I’d like to investigate the most common n-grams for each subreddit further. Third I would continue to work on a grid search to see if we can make this model stronger.

