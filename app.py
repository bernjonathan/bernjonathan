import pandas as pd
from nltk.corpus import wordnet
clear_df = pd.read_csv('testing.csv')
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
clear_df["clean_caption"] = clear_df["caption"].apply(lambda x: clean_text(x))
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(clear_df["clean_caption"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = clear_df.index
clear_df = pd.concat([clear_df, tfidf_df], axis=1)

head_df = pd.read_csv('head_df.csv')

predicting = head_df.append(clear_df, ignore_index = True, sort = False).\
               fillna(0).drop(['caption','clean_caption'], axis =1)
a = list(predicting.columns)
b = list(head_df.columns)
tracking = []
for element in a:
    if element not in b:
        tracking.append(element)
predicting = predicting.drop(tracking,axis = 1)
import pickle
with open(r"svm.pkl", "rb") as input_file:
    svm = pickle.load(input_file)
with open(r"knn.pkl", "rb") as input_file:
    knn = pickle.load(input_file)

y_pred = svm.predict(predicting)
clear_df['predict'] = y_pred[1:]
clear_df[['caption','predict']].to_csv('output_svm.csv',index = False, sep = ';')

y_pred = knn.predict(predicting)
clear_df['predict'] = y_pred[1:]
clear_df[['caption','predict']].to_csv('output_knn.csv',index = False, sep = ';')