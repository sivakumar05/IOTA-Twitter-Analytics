
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:47:34 2022

@author: smenta
"""

# Objective is to extract features of brand tweeter is talking about from tweet

'''

# Automated script to extract brand related features from tweet

 1) Steps involved
    a) Loading Data
    b) Clean the data
    c) Parse the sentence & identify features which are of NNP & having dependency type (compound,nsubj,appos,pobj)
 

 2) Assumption tweets are in english language
 
 3) Input files 
    1) tweets_companies.csv
    2) mapping_companies.csv
     

 4) Output file contains identified features from tweet
    1) Processed data.csv
 

'''

# Load required packages for analysis
#from nltk.corpus import words
import pandas as pd
import numpy as np
import nltk
import spacy
import re
import time
#import enchant

nlp = spacy.load("en_core_web_sm")

start_time = time.time()
#####################################################################
# Read data

# Input tweets data
filename='tweets_companies.csv' # file should contain fields - username,date,tweet
# Read data
maindata = pd.read_csv(filename)

# Input companies names mappings data
filename='mapping_companies.csv' # file should contain fields - username,corrected
# Read data
mappingdata = pd.read_csv(filename)

# EDA
maindata.head(2)
print('#.of records:',maindata.shape[0])
print("-----------------------")
# Tweets per group
print("Tweets per group")
maindata.groupby('username').size().reset_index(name='Observation')
print("-----------------------")

print("Tweets per Day")
#data.groupby(pd.to_datetime(data['date'], format='%Y%m%d')).size().reset_index(name='Observation')
maindata.groupby(pd.to_datetime(maindata['date'], format='%Y-%m-%d').dt.date).size().reset_index(name='Tweets')

# Data Cleaning


# remove user names
# case conversion
# POS Tagging

# function to remove emojis from text
def remove_emojis(data):
    '''
    This function removes emojis from text
    
    Input Parameters
    ----------
    data : String 
    Ex: "@RushHour_Pod Thanks for reaching out! Unfortunately, Coca-Cola Cinnamon was a limited time offer and will not return in 2021."

    Returns
    -------
    String 
    Ex: "@RushHour_Pod Thanks for reaching out! Unfortunately, Coca-Cola Cinnamon was a limited time offer and will not return in 2021.".

    '''
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# function to clean text data
def clean_text(data):
    '''
    This function cleans each tweet in the input dataframe
    
    Input Parameters
    ----------
    data : dataframe with following columns
    username	- datatype (object) contains brand related to tweet
    date	    - datatype (timestamp) contains timestamp of tweet
    tweet       - datatype (object) contains tweet text

    
    Returns
    -------
    data : dataframe with following columns
    username	- datatype (object) contains brand related to tweet
    date	    - datatype (timestamp) contains timestamp of tweet
    tweet       - datatype (object) contains tweet text
    tweet_cln   - datatype (object) contains cleaned tweet text

    '''
    
    # remove user names
    data['tweet_cln'] = [re.sub('@[\w]+','',x) for x in data['tweet']]

    # case conversion
    data['tweet_cln'] = [x.lower() for x in data['tweet_cln']]

    # Remove emojis
    
    data['tweet_cln'] = [remove_emojis(text) for text in data['tweet_cln']]

    # Remove url from text
    data['tweet_cln'] = [re.sub(r'http\S+', '', text) for text in data['tweet_cln']]

    # Remove # from text
    data['tweet_cln'] = [re.sub(r'#', '', text) for text in data['tweet_cln']]

    # Remove lead & trail spaces
    data['tweet_cln'] = [text.lstrip() for text in data['tweet_cln']]
    data['tweet_cln'] = [text.strip() for text in data['tweet_cln']]

    # Remove punctuation
    data['tweet_cln'] = data['tweet_cln'].map(lambda x: re.sub('[,\.!?]', '', x))
    return data




# Text data cleaning
data = clean_text(maindata)
data.head(2)
print(data.shape)
# Mapping revised username
data= data.merge(mappingdata, on='username', how='left')
data.head(2)
print(data.shape)

# Identify product is mentioned in tweet. If so, Product_flag=1; 0 
data['product_flag']= [ 1 if product.lower() in str(tweet) else 0 for tweet,product in zip(data['tweet_cln'],data['corrected'])]

# Remove space/characters between brand name with compound words
data['corrected1']=[x.lower().replace(" ", " ") for x in list(data['corrected'])]
data['corrected2']=[x.lower().replace("-", " ") for x in list(data['corrected'])]
data['corrected2']=[x.lower().replace(" ", "") for x in list(data['corrected2'])]
data['tweet_cln'] = [z.lower().replace(x, y) for x,y,z in zip(list(data['corrected1']),list(data['corrected2']),list(data['tweet_cln']))]
#data.to_csv('check.csv')
#####################################################################

#####################################################################
# Extract potential features from the cleaned tweet(text)
def extract_features(data):
    
    '''
    This function extracts features from cleaned tweet column in the input dataframe
    
    Input Parameters
    ----------
    data : dataframe with following columns
    username	- datatype (object) contains brand related to tweet
    date	    - datatype (timestamp) contains timestamp of tweet
    tweet       - datatype (object) contains tweet text
    tweet_cln       - datatype (object) contains cleaned tweet text
    corrected       - datatype (object) contains formatted username
    corrected1       - datatype (object) contains formatted username
    corrected2       - datatype (object) contains formatted username
    product_flag     - datatype (binary) 1 if tweet contains corrected username else 0
    
    Returns
    -------
    features : dataframe with following columns
    index - row number from input dataset
    username	- datatype (object) contains brand related to tweet
    date	    - datatype (timestamp) contains timestamp of tweet
    tweet       - datatype (object) contains tweet text
    tweet_cln       - datatype (object) contains cleaned tweet text
    corrected       - datatype (object) contains formatted username
    product_flag     - datatype (binary) 1 if tweet contains corrected username else 0
    feature          - datatype (object) contains extracted features from cleaned tweet text Ex: ['cinnamon']
    feature_category - datatype (object) contains feature categories of a tweet "identified/unidentified" Ex: "identified"
    
    '''
    
    # Drop columns 'corrected1', 'corrected2' from dataframe
    data = data.drop(['corrected1', 'corrected2'], axis = 1).reset_index()
    
    # Filter tweets product related to user name is not mentioned in tweet
    product_flg_0= data.loc[data['product_flag']==0]
    
    # Filter tweets product related to user name is  mentioned in tweet
    product_flg_1= data.loc[data['product_flag']==1]
    
    
    #import spacy
    nlp = spacy.load('en_core_web_sm')
    # Parse tweet to identify features
    features = [[index,token.text, token.tag_, token.head.text, token.dep_] for text,index in zip(product_flg_1['tweet_cln'],product_flg_1['index']) for token in nlp(text)]
    features=pd.DataFrame(features,columns=['index','text', 'tag_', 'head.text', 'dep_'])
    
    # Filter out words in tweets with tag_ =='NNP' and dependency of type ('compound','pobj','nsubj','appos')
    features_s = features.loc[(features['tag_']=='NNP') & (features['dep_'].isin(['compound','pobj','nsubj','appos']))]
    features_s.to_csv("Parsed data.csv")
    # List of words in product names
    lst=[y.lower() for x in list(set(data['corrected'])) for y in x.split("-")]
    lst=[y  for x in lst for y in x.split(" ")]
    
    #import nltk 
    nltk.download('words')
    words = nltk.corpus.words.words()
    
    # Filter out only english words from tweets
    features_p1 = features_s.loc[features_s['text'].isin(lst)]
    features_p1 = features_s.loc[features_s['head.text'].isin(words)]
    features_p1['text']=features_p1['head.text']
    #print("Test:",(set(features_p1['text'])) & set({'cinnamon'}))
    features_p2 = features_s.loc[~features_s['text'].isin(lst)]
    features_p2 = features_p2.loc[features_p2['text'].isin(words)]
    features_p2 = features_p2.loc[features_p2['dep_'].isin(words)]
    
    features_ss = pd.concat([features_p1,features_p2],axis=0)
    
    #print("Test:",(set(features_ss['text'])) & set({'cinnamon'}))
    # Identify of list of unrelated words from tweet of type 'NNP'
    f_count=[  [word,list(features_ss['text']).count(str(word))] for word in list(set(features_ss['text']))]
    
    f_count=pd.DataFrame(f_count,columns=['Word','count'])
    f_count=f_count.sort_values(by='count',ascending=False)
    #f_count.to_csv('word count.csv')
    
    # Omit words of length below 3
    f_count=f_count.loc[(f_count['Word']).str.len().gt(3)]
    
    
    # List of stopwords
    prepositions= ['about','above','across','after','against','along','among','around','at','before','behind','between','beyond','but','by','concerning','despite','down','during','except','following','for','from','in','including',	'into','like','near','of','off','on','onto','out',		'over' ,		'past'		,'plus','since',		'throughout','to','towards','under',		'until'		,'up','upon','up to','with','within','without']
    stopwords=['great','king','queen','prince','princess','south','north','east','west','twitter','facebook','january','febraury','march','april','may','june','july','august','september','october','november','december','sun','mon','tue','wed','thu','fri','sat' ]
    # Omit words which are in stopwords list
    f_count=f_count.loc[~f_count['Word'].isin(stopwords)]
    f_count=f_count.loc[~f_count['Word'].isin(prepositions)]
    
    # Get pos tag
    #f_count['tagged'] = [nltk.pos_tag(list(f_count['Word'])[word]) for word in range(f_count.shape[0])]
    # Omit words whose frequency is below 2
    f_count=f_count.loc[(f_count['count'])>2]

    # Remove irrelavant words based on their POS tags
    lst=list(f_count['Word'])
    f_count['tagged'] = [i[1]  for i in nltk.pos_tag(lst)]
    f_count = f_count.loc[~f_count['tagged'].isin(['FW','VBZ','VBD','VBG','JJ','IN','NNS'])]

    #f_count.to_csv('f_count.csv')
    
    # Identify list of features which met various conditions
    features_final = features_ss.loc[features_ss['text'].isin(list(f_count['Word']))]
    groups = pd.DataFrame(features_final.iloc[:,0:2].groupby('index').agg(lambda   x: list(set(list(x))))).reset_index()
      
    print(groups)
    
    # Map identified features to dataset corresponding to each tweet
    product_flg_1_f = product_flg_1.merge(groups, on='index', how='left')
    # Rename column text to feature
    product_flg_1_f.rename(columns = {'text':'feature'}, inplace = True)
    
    # Processing the tweets data sets where analysis product names are not mentioned in tweets
    product_flg_0_f = product_flg_0.copy()
    # create feature column with value '[]'
    product_flg_0_f['feature'] = '[]'
    
    # combine datasets
    features = pd.concat([product_flg_1_f,product_flg_0_f],axis=0)
    
    # Substitute null values with '[]' in feature column
    features['feature'] = np.where(features['feature'].isnull(),'[]',features['feature'])
    
    # Create feature_category column
    features['feature_category'] = np.where(features['feature']=='[]','unidentified','identified')
    
    return features

# Extract potential features from the cleaned tweet(text)
features = extract_features(data)
  

end_time = time.time()

print(["Total Run time:",end_time-start_time])
# Output data
features.to_csv("Processed data_v5.csv",index=False)


#####################################################################

#####################################################################


'''
# EDA

features.groupby('feature_category').size()

x=features.loc[features['feature_category']=='identified']
y=pd.DataFrame(x.groupby('feature').size()).reset_index()
y.columns=['feature','count']
y=y.sort_values(['count'],ascending=False)
y

# Validation

features.groupby('feature_category').size()
x=features.loc[features['feature_category']=='identified']
y=pd.DataFrame(x.groupby('feature').size()).reset_index()
y.columns=['f','c']
set(list(y['f']))


features.groupby('feature_category').size()
x=features.loc[features['feature_category']=='unidentified']
y=pd.DataFrame(x.groupby('feature').size()).reset_index()
y.columns=['f','c']
set(list(y['f']))
'''
#####################################################################