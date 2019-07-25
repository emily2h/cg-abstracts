import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()

tokenizer = Tokenizer(nlp.vocab)
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
lemmas = lemmatizer(u"functional", u"ADJ")
print(lemmas)
