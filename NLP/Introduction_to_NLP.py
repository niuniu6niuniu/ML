import nltk
import re

# - - -     1. Some regular expression examples     - - - #
# print(re.findall("[a-z]", "$34.33 cash."))
# print(re.findall("(name|phone):", "My name: Joe, my phone: (312)555-1212"))
# print(re.findall("([Ll]ion)s?", "Give it to Lions or the lion."))
# print(re.sub("[a-z]", "x", "Hey. I know this regex stuff..."))


# - - -     2. Text Processing     - - - #
# nltk.download()
# - Show how to access one of the gutenberg books included in NLTK
# print("gutenberg book ids-", nltk.corpus.gutenberg.fileids())
# - Load words from "Alice in Wonderland"
# alice = nltk.corpus.gutenberg.words("carroll-alice.txt")
# print("len(alice=", len(alice))
# print(alice[:100])
# - Load words from "Monty Python and the Holy Grail"
# grail = nltk.corpus.webtext.words("grail.txt")
# print("len(grail)-", len(grail))
# print(grail[:100])


# - - -     3. Plain Text Extraction     - - -  #
# - Extract plain text from non-plain text file(WORD,POWERPOINT,PDF,HTML,etc)
# - Word and Sentence Segmentation (Tokenization)
# - Code example: simple version of maxmatch algorithm for tokenization
def tokenize(str, dict):
    s = 0
    words = []
    while s < len(str):
        found = False

        # Find the biggest word in dict that matches str
        for word in dict:
            lw = len(word)
            if str[s:s+lw] == word:
                words.append(word)
                s += lw
                found = True
                break
        if not found:
            words.append(str[s])
            s += 1
    return words
# - Small dictionary of known words, longest words first
# dict = ["before", "table", "theta", "after", "where", "there", "bled", "said", "lead", "man", "her", "own", "the", "ran", "it"]
# # - This algorithm is designed to work with languages that don't have whitespace characters
# words1 = tokenize("themanranafterit", dict)   # works
# print(words1)
# words2 = tokenize("thetabledownthere", dict)   # fails
# print(words2)

# - Build-in NLTK tokenizer
# print(nltk.word_tokenize("the man, he ran after it's $3.23 dog on 03/23/2016."))
# print(nltk.sent_tokenize("The man ran after it. The table down there? Yes, down there!"))


# - - -     4. Stopword Removal     - - - #
# - Code example: simple algorithm for removing stopwords
stoppers = "a is of the this".split()

def removeStopwords(stopwords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stoppers])
    return newtxt
# newtxt = removeStopwords(stoppers, "this is a test of the stop word removal code.")
# print(newtxt)

# NLTK example: removing stopwords
from nltk.corpus import stopwords
stops = stopwords.words("English")
# print("len(stops)=", len(stops))
# print(removeStopwords(stops, "this is a test of the stop word removal code."))


# - - -     5. Text Normalization     - - - #
# - Case Removal
# str = " The man ran after it. The table down there? Yes, down there!"
# print(str.lower())

# - Stemming
# - NLTK example: stemming

def stem_with_porter(words):
    porter = nltk.PorterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words

def stem_with_lancaster(words):
    porter = nltk.LancasterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words

# str = "Please don't unbuckle your seat-belt while I am driving, he said"
#
# print("porter:", stem_with_porter(str.split()))
# print()
# print("lancaster: ", stem_with_lancaster(str.split()))


# - - -     6. Text Exploration     - - - #
# - NLTK example: frequency analysis
# import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist

# - get raw text from "Sense and Sensibility" by Jane Austin
raw = gutenberg.raw("austen-sense.txt")
fd_letters = FreqDist(raw)

words = gutenberg.words("austen-sense.txt")
fd_words = FreqDist(words)
sas = nltk.Text(words)

# - plot the frequency
# import matplotlib.pyplot as plt
# - letter frequency
# plt.figure(figsize=(20, 5))
# fd_letters.plot(100)
# - words frequency
# plt.figure(figsize=(20, 5))
# fd_words.plot(50)

# - Collocations (word pairs)
# print(sas.collocation_list())

# - Long words
# Let's look at the long words in the text
# longWords = [w for w in set(words) if len(w) > 13]
# print(longWords[:15])

# - Concordance Views (Keywords and the words that surrounded it)
# sas.concordance("affectionately")
# print()
# sas.concordance("correspondence")
# print()
# sas.concordance("dare")
# print()

# - Other Exploration tasks
# - Look at words similar to a word
# sas.similar("affection")

# Look at words as they appear over time in the book/document
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15, 4))
# sas.dispersion_plot(["sense", "love", "heart", "listen", " man", "woman"])


# - - -     7. Building Features     - - - #
# - Bag-of-Words (BOW)
# - Example Vocabulary: today, here, I, a, fine, sun, moon, bird, saw
# - Target sentence:    I saw a bird today.
# - BOW one-hot vector: 1 0 1 1 0 0 1 1

# - N-Grams
# - Unigrams, bigrams, trigrams

# - TD/IDF
# - TD: word frequency in the document (word count in document) / (total words in document)
# - IDF: weight of uniquess of word across all of the documents
# - TD-IDF: td_idf(t,d) = (wc(t,d) / wc(d)) / (dc(t) / dc())
#   wc(t,d): of occurences of term t in doc d
#   wc(d): of words in doc d
#   dc(t): of docs that contain at least 1 occurence of term t
#   dc(): of docs in collection

# - POS tagger
# import nltk
# print(nltk.pos_tag("they refuse to permit us to obtain the refuse permit".split()))

