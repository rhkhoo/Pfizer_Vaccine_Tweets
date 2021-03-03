# Pfizer_Vaccine_Tweets
 Using supervised and unsupervised learning in conjunction with NLP to explore tweets about the Pfizer COVID-19 vaccine <br>
![Wordcloud of Tweets about Pfizer COVID-19 Tweets](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/wordcloud.png)
<br>
This dataset is from [Kaggle](https://www.kaggle.com/gpreda/pfizer-vaccine-tweets), compiled by user Gabriel Preda. I will be clustering the tweets to find any common themes among them, determining the sentiment of each tweet to see how the general public has reacted to the Pfizer BioNTech vaccine, and finally analyzing how sentiment has changed over time.

# Data Cleaning

First, I remove any URLs and symbols from the tweets using Regular Expressions. Then, I convert the date column to a datetime object using pd.to_datetime().
> tweets['no_url'] = tweets['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0]) <br>
> tweets["alpha_num"] = tweets["no_url"].str.replace("[^a-zA-Z ]", "", regex=True)
 <br> 
 
I removed the stopwords included in NLTK's English stopwords plus a few that I added, like "vaccine," since these appear in virtually every tweet which would make clustering difficult. 

# Clustering
Using Tensorflow Hub's [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4), I encoded each tweet. DBSCAN and KMeans clustered the encoded tweets, and these are visualized using UMAP and PCA:
![Dimension-reduced visualization of DBSCAN clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/dbscan.png) <br>

![Dimension-reduced visualization of KMeans clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/kmeans.png)

<br>
The clusters are labeled using the word that occurs most frequently in each cluster. Some interesting labels include "today," "moderna," "dubai," and "ban." The tweets may vary by location since the clustering picked out a few location names.
<br>

# Sentiment Analysis

To determine the sentiment of each tweet, I used [Textblob's](https://textblob.readthedocs.io/en/dev/api_reference.html) Sentiment function.
>"The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a >float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
<br>
For this project, I will only be working with polarity.
<br>
The majority of tweets (57%) are neutral, followed by positive (33%), and the minority of tweet are negative (10%). Using these labels, I tried a couple of classifier models to see how we can predict a tweet's opinion of Pfizer's COVID-19 vaccine. RandomForestClassifier is able to predict sentiment with 86% accuracy; the model has difficulty distinguishing between Negative and Neutral. More training data would likely solve this as there is a class imbalance. KNeighborsClassifier is only able to achieve 60% accuracy, which is only achieved by the model predicting the majority class (neutral) for almost the entire test set.<br>
<br>

![Dimension-reduced visualization of KMeans clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/neutral.png) <br>
![Dimension-reduced visualization of KMeans clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/positive.png)<br>
![Dimension-reduced visualization of KMeans clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/negative.png)<br>
# Sentiment Over Time
I also wanted to see how opinions of the vaccine have changed over time
![Dimension-reduced visualization of DBSCAN clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/overtime.png) <br>
The trends of each type of tweet generally follow the same trend. There is a sharp decrease in all types of tweets on December 25th. Interestingly, the number of total tweets peaks on January 9th, the same day as the Capitol Riots. The sharp decline in the number of tweets approaching January 17th may be due to a data collection error. If the data were collected today, it is possible that this decrease would not be present.<br><br>
The Negative tweets have the youngest average account age of 5.75 years. Neutral tweets come from accounts with an average age of 6.15 years, and Positive tweets come from accounts with an average age of 6.96 years. However, the youngest account in each group do not follow this pattern. The Neutral group had the youngest account of only 5 minutes. 

# Conclusion
The majority of these tweets are neutral, and are likely news sources simply reporting updates on the vaccine. Whether or not the user is verified doesn't seem to matter. Pfizer and other companies with their own vaccines may find this information useful to address public concerns, as the quickest way to end the pandemic may well be to vaccinate everyone. <br><br>
To gauge the accuracy of TextBlob's sentiment function, it would be necessary to manually label each tweet as positive, negative, or neutral. Additionally, other classifiers such as SVC should be tried to get the best accuracy.<br><br>

It would also be interesting to work with the subjectivity aspect of the sentiment function to see if this affects how the tweets are labeled.<br><br>

Further work could also include looking for seasonality in the sentiment of tweets as more data is collected. Additionally, if more tweets are compiled, deep learning classification models could be built, possibly leading to even better accuracy.
