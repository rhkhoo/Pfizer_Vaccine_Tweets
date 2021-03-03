# Pfizer_Vaccine_Tweets
 Using supervised and unsupervised learning in conjunction with NLP to explore tweets about the Pfizer COVID-19 vaccine
![Wordcloud of Tweets about Pfizer COVID-19 Tweets](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/wordcloud.png)

This dataset is from [Kaggle]('https://www.kaggle.com/gpreda/pfizer-vaccine-tweets'), compiled by user Gabriel Preda. I will be clustering the tweets to find any common themes among them, determining the sentiment of each tweet to see how the general public has reacted to the Pfizer BioNTech vaccine, and finally analyzing how sentiment has changed over time.

# Data Cleaning

First, I remove any URLs and symbols from the tweets using Regular Expressions. Then, I convert the date column to a datetime object using pd.to_datetime().
> tweets['no_url'] = tweets['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
> tweets["alpha_num"] = tweets["no_url"].str.replace("[^a-zA-Z ]", "", regex=True)
 
I removed the stopwords included in NLTK's English stopwords plus a few that I added, like "vaccine," since these appear in virtually every tweet which would make clustering difficult. 

# Clustering
Using Tensorflow Hub's [Universal Sentence Encoder]('https://tfhub.dev/google/universal-sentence-encoder/4'), I encoded each tweet. DBSCAN and KMeans clustered the encoded tweets, and these are visualized using UMAP and PCA:
![Dimension-reduced visualization of DBSCAN clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/dbscan.png) <br>

![Dimension-reduced visualization of KMeans clustering](https://github.com/rhkhoo/Pfizer_Vaccine_Tweets/blob/main/Images/kmeans.png)
