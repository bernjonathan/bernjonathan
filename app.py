import pandas as pd
from nltk.corpus import wordnet
#membuka data csv menjadi data
clear_df = pd.read_csv('testing.csv')
#membuat data menjadi jenis katanya dalam bahasa inggris
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
##preprocessing text
def clean_text(text):
    # mengecilkan semua huruf agar seragam "lowercas"
    text = text.lower()
    # memisahkan kata menjadi satu kata dan menghapus tanda baca "tokenize dan punctuatuon"
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # mengahapus kata yang merupakan digit
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # menghilangkan kata yang tidak penting seperti is, a, dll "stopwords"
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens untuk lemmatize
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # membuat menjadi kata dasar "lemmatize"
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# menaruh data bersih di dataframe colom clean_caption
clear_df["clean_caption"] = clear_df["caption"].apply(lambda x: clean_text(x))
from sklearn.feature_extraction.text import TfidfVectorizer
#membuat vectorizer dengan tfidf
tfidf = TfidfVectorizer(min_df = 1)
tfidf_result = tfidf.fit_transform(clear_df["clean_caption"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = clear_df.index
clear_df = pd.concat([clear_df, tfidf_df], axis=1)

#memotong vector kata dan menambahkan kata agar sesuai dengan yang mesin hitung
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

#mengimport mesin 
import pickle
with open(r"svm.pkl", "rb") as input_file:
    svm = pickle.load(input_file)
with open(r"knn.pkl", "rb") as input_file:
    knn = pickle.load(input_file)
#memprediksi mesin dengan svm
y_pred = svm.predict(predicting)
clear_df['predict'] = y_pred[1:]

test = pd.DataFrame(clear_df.predict.value_counts().reset_index()\
                    .rename(columns={'index':'caption'}))
#mengeluarkan hasil prediksi svm di csv
output_df = clear_df.append(pd.Series(), ignore_index=True)
output_df = pd.concat([output_df[['caption','predict']],test])
output_df[['caption','predict']].to_csv('output_svm.csv',index = False, sep = ';')

print(output_df)
#memprediksi mesin dengan knn
y_pred = knn.predict(predicting)
clear_df['predict'] = y_pred[1:]
test = pd.DataFrame(clear_df.predict.value_counts().reset_index()\
                    .rename(columns={'index':'caption'}))
#mengeluarkan hasil prediksi knn di csv
output_df = output_df.append(pd.Series(), ignore_index=True)
output_df = pd.concat([clear_df[['caption','predict']],test])
output_df[['caption','predict']].to_csv('output_knn.csv',index = False, sep = ';')
