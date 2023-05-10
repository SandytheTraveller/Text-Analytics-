# downloading all necessary resources
import nltk
# nltk.download('all')

# load the Brown Corpus
from nltk.corpus import brown
print('Total Categories: ', len(brown.categories()))

print(brown.categories())

# tokenized sentences
print(brown.sents(categories='mystery'))

# POS tagged sentences
print(brown.tagged_sents(categories='mystery'))

# get sentences in natural form
sentences = brown.sents(categories='mystery')
sentences = [' '.join(sentence_token) for sentence_token in sentences]
print(sentences[0:5]) # printing first 5 sentences

# get tagged words
tagged_words = brown.tagged_words(categories='mystery')

# get nouns from tagged words
nouns = [(word, tag) for word, tag in tagged_words if any(noun_tag in tag for noun_tag in ['NP', 'NN'])]

# print the first 10 nouns
print(nouns[0:10])

# build frequency distribution for nouns
nouns_freq = nltk.FreqDist([word for word, tag in nouns])

# print top 10 occurring nouns
print(nouns_freq.most_common(10))

# Accessing the Reuters Corpus
# load the Reuters Corpus
from nltk.corpus import reuters

print('Total Categories: ', len(reuters.categories()))
print(reuters.categories())

# get sentences in housing and income cateories
sentences = reuters.sents(categories=['housing', 'income'])
sentences = [' '.join(sentence_tokens) for sentence_tokens in sentences]
print(sentences[0:5])

# file id based access
print(reuters.fileids(categories=['housing', 'income']))

print(reuters.sents(fileids=[u'test/16118', u'test/18534']))

# accessing the WorldNet Corpus
# load the wordnet Corpus
from nltk.corpus import wordnet as wn

word = 'hike' # taking hike as our word of interest

# get word synsets
word_sysnets = wn.synsets(word)
print(word_sysnets)

# get details for each synonym in synset
for synset in word_sysnets:
    print('Synset Name: ', synset.name())
    print('POS Tag: ', synset.pos())
    print('Definition: ', synset.definition())
    print('Examples: ', synset.examples())
    print()

