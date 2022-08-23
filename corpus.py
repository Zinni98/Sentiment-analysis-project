from nltk.corpus import movie_reviews
import numpy as np
import torch

class MovieReviewsCorpus():
    def __init__(self, preprocess_pipeline):
        # list of documents, each document is a list containing words of that document
        self.mr = movie_reviews
        self.unprocessed_corpus = self._get_corpus()
        self.flattened_corpus, self.labels = self._flatten()
        self.corpus_words = self.get_corpus_words()
        self.vocab = self._create_vocab()


    def _list_to_str(self, doc) -> str:
        """
        Put all elements of the list into a single string, separating each element with a space.
        """
        return " ".join([w for sent in doc for w in sent])


    def _flatten(self):
        """
        Returns
        -------
        list[list[str]]
            Each inner list represents a document. Each document is a list of tokens.
        """

        # 3 nested list: each list contain a document, each inner list contains a phrase (until fullstop), each phrase contains words.
        neg = self.mr.paras(categories = "neg")
        pos = self.mr.paras(categories = "pos")

        corpus = [[w for w in self._list_to_str(d).split(" ")] for d in pos] + [[w for w in self._list_to_str(d).split(" ")] for d in neg]
        labels = [0] * len(pos) + [1] * len(neg)
        return corpus, labels

    def _get_corpus(self):
        neg = self.mr.paras(categories = "neg")
        pos = self.mr.paras(categories = "pos")

        return neg + pos

    def movie_reviews_dataset_raw(self):
        """
        Returns the dataset containing:

        - A list of all the documents
        - The corresponding label for each document

        Returns
        -------
        tuple(list, list)
            The dataset: first element is the list of the document, the second element of the tuple is the associated label (positive or negative) for each document
        """

        return self.corpus, self.labels

    def get_sentence_ds(self):
        neg = self.mr.paras(categories = "neg")
        pos = self.mr.paras(categories = "pos")

        pos = [phrase for doc in pos for phrase in doc]
        neg = [phrase for doc in neg for phrase in doc]

        labels = np.array([0] * len(pos) + [1] * len(neg))
        corpus = neg+pos
        return corpus, labels


    def get_corpus_words(self) -> list:
        return [w for doc in self.corpus for w in doc]
    
    def get_embedding_matrix(self, embedding, embedding_dim):
        """
        Returns
        -------
        np.ndarray
            A 2D which each row has the corresponding embedding from the vocabulary
        """
        matrix_length = len(self.vocab)
        embedding_matrix = np.zeros(matrix_length, embedding_dim)
        for idx, key in enumerate(self.vocab.keys()):
            try:
                embedding_matrix[idx] = embedding[key]
            except:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size = (embedding_dim, ))
        return embedding_matrix
    
    def get_indexed_representation(self):
        """
        Returns
        -------
        Dictionary
            Containing correspondences word -> index
        
        list(list(torch.tensor))
            The corpus represented as indexes corresponding to each word
        """
        vocab = {}
        for idx, key in enumerate(self.vocab.keys()):
            vocab[key] = idx
        
        indexed_corpus = [[torch.tensor(vocab[w], dtype=torch.int32) for w in doc] for doc in self.corpus]
        return indexed_corpus


    def _create_vocab(self):
        vocab = dict()
        for word in self.corpus_words:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1
        return vocab



    def __len__(self):
        return len(self.corpus)

