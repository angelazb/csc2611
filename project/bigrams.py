import pandas as pd
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
import numpy as np
from statistics import mean

# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
#from textblob import TextBlob
import contractions
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def get_token_words(survey_data, col_name, stopwords_list, title):
    """ Return a list of all the token words that we can use to
    generate the bigrams.

    survey_data: Pandas data frame, contains data to analyze
    col_name: Name of specific column to analyze responses
    stopwords_list: List of words to ignore that might be in the question
    title: Title of sentiment
    """
    # Drop null values, reset the index
    print(survey_data.columns)
    data = survey_data.dropna(subset=[col_name])
    data = data.reset_index(drop=True)

    # Get only the column that you need
    responses_data = data[col_name]
    # Make it a list without the col name
    responses = []
    for i in range(len(responses_data) - 1):
        responses.append(str(responses_data[i + 1]))
    
    sentiment_list = sentiment(responses)

    sent = []

    for i in range(len(sentiment_list)):
        if sentiment_list[i] != "n/a":
            if sentiment_list[i] > 0:
                sent.append("pos")
            elif sentiment_list[i] == 0:
                sent.append("neutral")
            else:
                sent.append("neg")
        else:
            sent.append("n/a")

    sentiment_table = pd.DataFrame({'sentiment' : sentiment_list})
    sentiment_table['sent_word'] = sent
    sentiment_table['responses'] = responses

    sentiment_table.to_csv("files/Neut_"+title, index=False)

    processed = []
    index = 0
    while index < len(responses):
        # Look at words in one response
        # lowercase
        responsewords = responses[index].lower()

        # remove punctuation
        responsewords = re.sub('[!#?,.:";\']', "", responsewords)

        # split into a list
        resultwords = responsewords.split()

        # expand contractions
        for i in range(len(resultwords)):
            word = resultwords[i]
            resultwords[i] = contractions.fix(word)
        
        # remove stopwords
        for word in stopwords_list:
            if word in resultwords:
                resultwords.remove(word)
        
        # Joing back as text
        processed.append(" ".join(resultwords))
        index += 1
    
    # Join the responses into one big text
    text = " ".join(processed)

    # Stemming: removes suffices: ing, ly, s
    # Lemmatization (root word)
    st = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text_stem = []
    for word in text.split(" "):
        word = st.stem(word)
        text_stem.append(lemmatizer.lemmatize(word))
    text = " ".join(text_stem)

    # Passing the string text into word tokenize for breaking the sentences
    tokens = word_tokenize(text)

    # Word cloud
    #wordcloud = WordCloud(background_color="white").generate(text)
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

    # Removing english stopwords
    eng_stopwords = set(stopwords.words("english"))

    tokens = [w for w in tokens if w not in eng_stopwords]

    return tokens

def sentiment(responses):
    """ Return a list of sentiment scores given the responses

    responses: list of students' responses.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for response in responses:
        if response != "":
            score = [sia.polarity_scores(response)["compound"] for sentence in nltk.sent_tokenize(response)]
            if score != []:
                sentiment_scores.append(mean(score))
            else:
                sentiment_scores.append("n/a")
        else:
            sentiment_scores.append("n/a")
    return sentiment_scores

def create_bigrams(tokens):
    """ Return a list of biagrams based on the given tokens

    tokens: NLTK tokens returned by get_tokens
    """
    bigrams = list(nltk.bigrams(tokens))
    # Merge duplicated bigrams (i.e. (a, b), (b, a))
    bigrams = [tuple(sorted(b)) for b in bigrams]
    # Remove bigrams that have same token twice
    bigrams = [b for b in bigrams if b[0] != b[1]]

    return bigrams



def freq_csv(title, tokens, col1, col2, bigram_dist=False):
    """ Export the frequency distribution produced by NLTK
    as a CSV file. Option for normal frequency distribution, or bigram
    frequency distribution.
    
    title: name of file
    tokens: NLTK tokens returned by get_tokens
    col1: name to first column
    col2: name to second column
    bigram_dist: Boolean, True if bigram is wanted, false otherwise
    """
    if not bigram_dist:
        fdist = nltk.FreqDist(tokens)

    else:
        bigrams = create_bigrams(tokens)
        # Compute frequency distribution for all the bigrams in the text
        fdist = nltk.FreqDist(bigrams)

    # Turn fdist dictionary into data frame, dictionary keys
    # becomes index (so only 1 column, values)
    df_fdist = pd.DataFrame.from_dict(fdist, orient='index')

    # Name values column, sort descending
    df_fdist.columns = [col2]
    df_fdist = df_fdist.sort_values(by=col2, ascending=False)

    # Turn index into column
    df_fdist = df_fdist.reset_index()
    df_fdist.columns = [col1, col2]

    col3 = []
    n = len(df_fdist[col1])
    n_count = np.sum(df_fdist[col2])
    print(n_count)
    for i in range(n):
        col3.append(round(df_fdist[col2][i]/n_count, 4))
    df_fdist["normalized_count"] = col3
    file = title + '.csv'
    df_fdist.to_csv(file, index=False)


    

def bigram_graph(tokens, numwords, words_title):
    """Using tokens, output word frequency and bigram plots
    
    tokens: Ordered list of word tokens
    numwords: Int, number of entries for x-axis
    words_title: Title of graph
    """
    # Finding the frequency distinct in the tokens
    fdist = FreqDist(tokens)

    total = fdist.N()
    for word in fdist:
        fdist[word] /= float(total)

    fdist.plot(numwords, cumulative=False, title=words_title)
    # fdist.tabulate()

    # Create  bigrams
    bigrams = create_bigrams(tokens)

    # Compute frequency distribution for all the bigrams in the text
    fdist_bgs = nltk.FreqDist(bigrams)

    total2 = fdist_bgs.N()
    for word in fdist_bgs:
        fdist_bgs[word] /= float(total2)
    
    fdist_bgs.plot(numwords, cumulative=False, title=words_title)
    # collocations = nltk.collocations()
    # fdist_clc = nltk.FreqDist(collocations)

    # fdist_clc.plot(numwords, cumulative=False, title=bigram_title)

'''
Winter Prepare: 2513 Words and 2480 Bigrams
Winter Prepare - more stop words: 2306 Words and 2275 Bigrams
'''
filename = "3 - Winter 2020 Mid.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")
tokens = get_token_words(df, "Q235", ["PCRS", "start", "prepare", "'"], "Winter_Prepare_Sentiment.csv")
#freq_csv(title="files/Winter_Prepare_Start_Words", tokens=tokens, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Winter_Prepare_Start_Words_Bigrams", tokens=tokens, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Prepare early - Winter 2020")
tokens_more = get_token_words(df, "Q235", ["PCRS", "start", "prepare","starting", "course", "courses", "work", "'"], "Winter_Prepare_Sentiment_More.csv")
#freq_csv(title="files/Winter_Prepare_Start_Words_More", tokens=tokens_more, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Winter_Prepare_Start_Words_More_Bigrams", tokens=tokens_more, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens_more, 20, "Barriers to start Prepare early - Winter 2020\nExcluding most frequent words")

'''
Winter Perform: 2555 Words and 2526 Bigrams
Winter Perform - more stop words: 2363 Words and 2335 Bigrams
'''
filename = "3 - Winter 2020 Mid.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")
tokens = get_token_words(df, "Q236", ["PCRS", "start", "starting",  "perform", "\'", "\"'\""], "Winter_Perform_Sentiment.csv")
#freq_csv(title="files/Winter_Perform_Start_Words", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Winter_Perform_Start_Words_Bigrams", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Perform early - Winter 2020")
tokens_more = get_token_words(df, "Q236", ["PCRS", "start", "starting",  "perform",  "course", "courses", "work", "\'", "\"'\""], "Winter_Perform_Sentiment_More.csv")
#freq_csv(title="files/Winter_Perform_Start_Words_More", tokens=tokens_more, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Winter_Perform_Start_Words_More_Bigrams", tokens=tokens_more, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens_more, 20, "Barriers to start Perform early - Winter 2020\nExcluding most frequent words")

'''
Fall Prepare: 7079 Words and 7042 Bigrams
Fall Prepare - more stop words: 6406 Words and 6378 Bigrams
'''
filename = "5 - Fall 2020 Mid.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")
tokens = get_token_words(df, "Q235", ["PCRS", "start", "prepare", "'"], "Fall_Prepare_Sentiment.csv")
#freq_csv(title="files/Fall_Prepare_Start_Words", tokens=tokens, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Fall_Prepare_Start_Words_Bigrams", tokens=tokens, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Prepare early - Fall 2020")
tokens_more = get_token_words(df, "Q235", ["PCRS", "start", "prepare","starting","course", "work","courses", "'"], "Fall_Prepare_Sentiment_More.csv")
#freq_csv(title="files/Fall_Prepare_Start_Words_More", tokens=tokens_more, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Fall_Prepare_Start_Words_More_Bigrams", tokens=tokens_more, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens_more, 20, "Barriers to start Prepare early - Fall 2020\nExcluding most frequent words")

'''
Fall Perform: 7100 Words and 7072 Bigrams
Fall Perform - more stop words: 6560 Words and 6536 Bigrams
'''
filename = "5 - Fall 2020 Mid.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")
tokens = get_token_words(df, "Q236", ["PCRS", "start", "starting", "perform", "\'", "\"'\""], "Fall_Perform_Sentiment.csv")
#freq_csv(title="files/Fall_Perform_Start_Words", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Fall_Perform_Start_Words_Bigrams", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Perform early - Fall 2020")
tokens_more = get_token_words(df, "Q236", ["PCRS", "start", "starting",  "perform",  "course", "courses", "work", "\'", "\"'\""], "Fall_Perform_Sentiment_More.csv")
#freq_csv(title="files/Fall_Perform_Start_Words_More", tokens=tokens_more, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Fall_Perform_Start_Words__More_Bigrams", tokens=tokens_more, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens_more, 20, "Barriers to start Perform early - Fall 2020\nExcluding most frequent words")

'''
Both Prepare: 9612 Words and 9543 Bigrams
Both Prepare - more stop words: 8731 Words and 8673 Bigrams
'''
filename = "Fall and Winter 2020.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")
tokens = get_token_words(df, "ï»¿Q235", ["PCRS", "start", "prepare","'"], "Both_Prepare_Sentiment.csv")
#freq_csv(title="files/Both_Prepare_Start_Words", tokens=tokens, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Both_Prepare_Start_Words_Bigrams", tokens=tokens, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Prepare early - Both 2020")
tokens_more = get_token_words(df, "ï»¿Q235", ["PCRS", "start", "prepare","starting", "course", "work","courses","'"], "Both_Prepare_Sentiment_More.csv")
#freq_csv(title="files/Both_Prepare_Start_Words_More", tokens=tokens_more, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Both_Prepare_Start_Words_More_Bigrams", tokens=tokens_more, col1="prepare_start_bigram", col2="prepare_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens_more, 20, "Barriers to start Prepare early - Both 2020\nExcluding most frequent words")

'''
Both Perform: 9670 Words and 9614 Bigrams
Both Perform - more stop words: 8938 Words and 8887 Bigrams
'''
filename = "Fall and Winter 2020.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")
tokens = get_token_words(df, "Q236", ["PCRS", "start", "starting",  "perform", "\'", "\"'\""], "Both_Perform_Sentiment.csv")
#freq_csv(title="files/Both_Perform_Start_Start_Words", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Both_Perform_Start_Start_Words_Bigrams", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Perform early - Both 2020")
tokens = get_token_words(df, "Q236", ["PCRS", "start", "starting",  "perform", "course", "courses", "work", "\'", "\"'\""], "Both_Perform_Sentiment_More.csv")
#freq_csv(title="files/Both_Perform_Start_Start_Words_More", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=False)
#freq_csv(title="files/Both_Perform_Start_Start_Words_More_Bigrams", tokens=tokens, col1="perform_start_bigram", col2="perform_start_bigram_freq", bigram_dist=True)
#bigram_graph(tokens, 20, "Barriers to start Perform early - Both 2020\nExcluding most frequent words")
