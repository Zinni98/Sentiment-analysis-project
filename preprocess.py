from cmath import phase
from dis import findlabels
from unicodedata import name
from nltk.corpus import stopwords
import re
import spacy


CONTRACTION_MAP =  {"ain't": "is not",
                        "aren't": "are not",
                        "can't": "cannot",
                        "can't've": "cannot have",
                        "'cause": "because",
                        "could've": "could have",
                        "couldn't": "could not",
                        "couldn't've": "could not have",
                        "didn't": "did not",
                        "doesn't": "does not",
                        "don't": "do not",
                        "hadn't": "had not",
                        "hadn't've": "had not have",
                        "hasn't": "has not",
                        "haven't": "have not",
                        "he'd": "he would",
                        "he'd've": "he would have",
                        "he'll": "he will",
                        "he'll've": "he he will have",
                        "he's": "he is",
                        "how'd": "how did",
                        "how'd'y": "how do you",
                        "how'll": "how will",
                        "how's": "how is",
                        "i'd": "i would",
                        "i'd've": "i would have",
                        "i'll": "i will",
                        "i'll've": "i will have",
                        "i'm": "i am",
                        "i've": "i have",
                        "isn't": "is not",
                        "it'd": "it would",
                        "it'd've": "it would have",
                        "it'll": "it will",
                        "it'll've": "it will have",
                        "it's": "it is",
                        "let's": "let us",
                        "ma'am": "madam",
                        "mayn't": "may not",
                        "might've": "might have",
                        "mightn't": "might not",
                        "mightn't've": "might not have",
                        "must've": "must have",
                        "mustn't": "must not",
                        "mustn't've": "must not have",
                        "needn't": "need not",
                        "needn't've": "need not have",
                        "o'clock": "of the clock",
                        "oughtn't": "ought not",
                        "oughtn't've": "ought not have",
                        "shan't": "shall not",
                        "sha'n't": "shall not",
                        "shan't've": "shall not have",
                        "she'd": "she would",
                        "she'd've": "she would have",
                        "she'll": "she will",
                        "she'll've": "she will have",
                        "she's": "she is",
                        "should've": "should have",
                        "shouldn't": "should not",
                        "shouldn't've": "should not have",
                        "so've": "so have",
                        "so's": "so as",
                        "that'd": "that would",
                        "that'd've": "that would have",
                        "that's": "that is",
                        "there'd": "there would",
                        "there'd've": "there would have",
                        "there's": "there is",
                        "they'd": "they would",
                        "they'd've": "they would have",
                        "they'll": "they will",
                        "they'll've": "they will have",
                        "they're": "they are",
                        "they've": "they have",
                        "to've": "to have",
                        "wasn't": "was not",
                        "we'd": "we would",
                        "we'd've": "we would have",
                        "we'll": "we will",
                        "we'll've": "we will have",
                        "we're": "we are",
                        "we've": "we have",
                        "weren't": "were not",
                        "what'll": "what will",
                        "what'll've": "what will have",
                        "what're": "what are",
                        "what's": "what is",
                        "what've": "what have",
                        "when's": "when is",
                        "when've": "when have",
                        "where'd": "where did",
                        "where's": "where is",
                        "where've": "where have",
                        "who'll": "who will",
                        "who'll've": "who will have",
                        "who's": "who is",
                        "who've": "who have",
                        "why's": "why is",
                        "why've": "why have",
                        "will've": "will have",
                        "won't": "will not",
                        "won't've": "will not have",
                        "would've": "would have",
                        "wouldn't": "would not",
                        "wouldn't've": "would not have",
                        "y'all": "you all",
                        "y'all'd": "you all would",
                        "y'all'd've": "you all would have",
                        "y'all're": "you all are",
                        "y'all've": "you all have",
                        "you'd": "you would",
                        "you'd've": "you would have",
                        "you'll": "you will",
                        "you'll've": "you will have",
                        "you're": "you are",
                        "you've": "you have",
                    }
class MRAbstractPipeline():
    def __init__(self):
        self.pipeline = []
    
    def pipe(self, corpus):
        for el in self.pipeline:
            corpus = el(corpus)
        return corpus
    
    def __call__(self, *args, **kwds):
        if args[0] == None:
            raise ValueError("Need a corpus as argument")
        corpus = args[0]
        return self.pipe(corpus)
        

class MRPipelineTokens(MRAbstractPipeline):
    """
    Pipeline for documents represented as list of tokens
    """
    def __init__(self):
        super(MRPipelineTokens, self).__init__()
        self.pipeline = [self.remove_underscores, 
                         self.reducing_character_repetitions,
                         self.clean_contractions,
                         self.clean_special_chars,
                         self.remove_stop_words]

    def remove_underscores(self, corpus):
        """
        Solves the problem where some of the words are surrounded by underscores
        (e.g. "_hello_")
        """
        for doc in corpus:
            for idx, word in enumerate(doc):
                if "_" in word:
                    cleaned_word = self._clean_word(word)
                    doc[idx] = cleaned_word
        return corpus


    def _clean_word(self, word: str):
        word = word.replace("_", " ")
        # remove spaces before and after the word
        word = word.split()
        word = " ".join(word)
        return word
    
    def reducing_character_repetitions(self, corpus):
        
        new_corpus = []
        for doc in corpus:
            new_doc = [self._clean_repetitions(w) for w in doc]
            new_corpus.append(new_doc)
        return new_corpus

    # inspired by https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a
    def _clean_repetitions(self, word):
        """
        This Function will reduce repetition to two characters 
        for alphabets and to one character for punctuations.

        Parameters
        ----------
            word: str                
        Returns
        -------
        str
            Finally formatted text with alphabets repeating to 
            one characters & punctuations limited to one repetition 
            
        Example:
        Input : Realllllllllyyyyy,        Greeeeaaaatttt   !!!!?....;;;;:)
        Output : Really, Great !?.;:)

        """
        # Pattern matching for all case alphabets
        pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)

        # Limiting all the repetitions to two characters.
        # MODIFIED: keep only one repetition of the character
        formatted_text = pattern_alpha.sub(r"\1\1", word) 

        # Pattern matching for all the punctuations that can occur
        pattern_punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')

        # Limiting punctuations in previously formatted string to only one.
        combined_formatted = pattern_punct.sub(r'\1', formatted_text)

        # The below statement is replacing repetitions of spaces that occur more than two times with that of one occurrence.
        final_formatted = re.sub(' {2,}',' ', combined_formatted)
        return final_formatted
    
    def clean_contractions(self, corpus):
        new_corpus = []
        for doc in corpus:
            new_doc = []
            for word in doc:
                try:
                    correct = CONTRACTION_MAP[word]
                    correct = correct.split()
                    new_doc += correct
                except:
                    new_doc.append(word)
            new_corpus.append(new_doc)
        return new_corpus

    def clean_special_chars(self, corpus):
        new_corpus = [[self._clean_special_word(w) for w in doc] for doc in corpus] 
        return new_corpus
    
    def _clean_special_word(self, word):
        # The formatted text after removing not necessary punctuations.
        formatted_text = re.sub(r"[^a-zA-Z0-9:€$-,%.?!]+", '', word) 
        # In the above regex expression,I am providing necessary set of punctuations that are frequent in this particular dataset.
        return formatted_text
    
    def remove_stop_words(self, corpus):
        stops = stopwords.words("english")
        stops = [word for word in stops if "'t" not in word or "not" not in word]
        return [[word for word in doc if word not in stops] for doc in corpus]
    

class MRPipelinePhrases(MRAbstractPipeline):
    """
    Pipeline for documents represented as list of phrases
    """
    def __init__(self):
        super(MRPipelinePhrases, self).__init__()
        self.pipeline = [self.remove_underscores, 
                         self.clean_special_chars,
                         self.reducing_character_repetitions,
                         self.lemmatize]

    def remove_underscores(self, corpus):
        """
        Solves the problem where some of the words are surrounded by underscores
        (e.g. "_hello_")
        """
        new_corpus = [self._clean_word(doc) for doc in corpus]
        return new_corpus


    def _clean_word(self, doc: str):
        doc = doc.replace("_", " ")
        return doc
    
    def reducing_character_repetitions(self, corpus):
        new_corpus = [self._clean_repetitions(doc) for doc in corpus]
        return new_corpus
    
    # inspired by https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a
    def _clean_repetitions(self, word):
        """
        This Function will reduce repetition to two characters 
        for alphabets and to one character for punctuations.

        Parameters
        ----------
            word: str                
        Returns
        -------
        str
            Finally formatted text with alphabets repeating to 
            one characters & punctuations limited to one repetition 
            
        Example:
        Input : Realllllllllyyyyy,        Greeeeaaaatttt   !!!!?....;;;;:)
        Output : Realy, Great !?.;:)

        """
        # Pattern matching for all case alphabets
        pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)

        # Limiting all the repetitions to two characters.
        # MODIFIED: keep only one repetition of the character
        formatted_text = pattern_alpha.sub(r"\1\1", word) 

        # Pattern matching for all the punctuations that can occur
        pattern_punct = re.compile(r'([., /#!$%^&*?;:{}=_`~()+-])\1{1,}')

        # Limiting punctuations in previously formatted string to only one.
        combined_formatted = pattern_punct.sub(r'\1', formatted_text)

        # The below statement is replacing repetitions of spaces that occur more than two times with that of one occurrence.
        final_formatted = re.sub(' {2,}',' ', combined_formatted)
        return final_formatted

    def clean_special_chars(self, corpus):
        new_corpus = [self._clean_special_word(doc)  for doc in corpus]
        return new_corpus
    
    def _clean_special_word(self, word):
        # The formatted text after removing not necessary punctuations.
        formatted_text = re.sub(r"[^a-zA-Z0-9:€$-,%.?!]+", ' ', word) 
        # In the above regex expression,I am providing necessary set of punctuations that are frequent in this particular dataset.
        return formatted_text
    

    def lemmatize(self, corpus):
        nlp = spacy.load('en_core_web_sm')
        return [[token.lemma_ for token in nlp(doc)] for doc in corpus]
        

if __name__ == "__main__":
    example = [["_Lorrrrem_::-", "ipsum", "mustn't"], ["consectetur__", "§adipiçç@scing"], ["seddddd", "_eiu_smod_", "tempor", "incididunt"]]

    example_p = [["_Lorrrrem_::- ipsum mustn't sit amet", " consectetur__ §adipiçç@scing", "seddddd do _eiu_smod_"], ["frequent in]]] this particular dataset."]]

    nlp = spacy.load('en_core_web_sm')
    for token in nlp("Don't do that"):
        print(token.lemma_)