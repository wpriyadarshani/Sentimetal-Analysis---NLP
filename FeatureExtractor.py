#Author - Wajira Abeysinghe

#https://www.linkedin.com/pulse/text-classification-using-bag-words-approach-nltk-scikit-rajendran/?fbclid=IwAR0q-qbzDPU8JdOMIvSiN5tNakpdAxlRbYqFSfD_od9ns_Gui1_9KS5nBUo
#https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk?fbclid=IwAR31k9IneP0j6sL-8URtc8qmhmdrAPXBCL9jHPN0Pn4Snn2mAAo84eB-OVU
#https://stackabuse.com/text-classification-with-python-and-scikit-learn/


import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import collections
import sys
reload(sys)
sys.setdefaultencoding('utf8')


nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from autocorrect import spell

from nltk.stem import PorterStemmer
porter = PorterStemmer()



data = pd.read_csv("data.csv", header=None, engine = 'python', encoding = "ISO-8859-1")
data = data.values

np.random.shuffle(data)
data = pd.DataFrame(data)
text = data.iloc[:10000, -1].values

print(data.columns)
# text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
data = data.values
y = data[:10000, 0]
c = collections.Counter(y)

#show the bar graph of classes
plt.bar(c.keys(), c.values())
plt.xlabel('labels')
plt.ylabel('No of data')
# /plt.show()
plt.savefig("graph.png")

count = 0
for line in text:
	count=count+1
	print("---------------------------------------------------------------------------- \n", count)
	print("original --- > " , line)
	line = re.sub('[^A-Za-z]', ' ', line) #remove non-alphabetic character
	line = re.sub(r'(?:^| )\w(?:$| )', ' ', line).strip() #remove single character
	line = re.sub(r"http\S+", "", line)#remove hyperlinks

	line =  line.lower() #convert to lower case
	tokenized_word=word_tokenize(line)
	for word in tokenized_word:
	    if (word in stop_words):
	        tokenized_word.remove(word)

	
	for i in range(len(tokenized_word)):
		print("________________________________________________ \n")
		print(" initial world----> ", tokenized_word[i])
		tokenized_word[i] = porter.stem(tokenized_word[i])
		print(" stemming---------> ", tokenized_word[i])
		tokenized_word[i] = lem.lemmatize(tokenized_word[i],"v")
		tokenized_word[i] = lem.lemmatize(tokenized_word[i],"n")
		print(" lemmatization----> ", tokenized_word[i])
		tokenized_word[i] = spell(tokenized_word[i])
		print(" spell checker----> ", tokenized_word[i])
		print("________________________________________________ \n")


	print("updated -->", tokenized_word)



# from sklearn.feature_extraction.text import CountVectorizer
# # matrix = CountVectorizer(max_features=-1, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# matrix = CountVectorizer()
# X = matrix.fit_transform(text).toarray()

# print(matrix.get_feature_names())


# print(X.shape)

# df = pd.DataFrame(X)
# print(X)
# df.to_csv("output.csv", sep='\t')


from sklearn.feature_extraction.text import TfidfVectorizer
# tfidfconverter = TfidfVectorizer(max_features=-1, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
tfidfconverter = TfidfVectorizer()
X = tfidfconverter.fit_transform(text).toarray()
print(tfidfconverter.vocabulary_)
# print(X[0])
features = tfidfconverter.get_feature_names()

print(X.shape)

df = pd.DataFrame(X)
df.columns = features
df['labels'] = y
df = df[['labels'] + df.columns[:-1].tolist()]

print(df)
df.to_csv("output_tfid.csv", sep='\t')
