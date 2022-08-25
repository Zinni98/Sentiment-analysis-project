from nltk.corpus import movie_reviews
import numpy as np
import torch

class MovieReviewsCorpus():
    def __init__(self, preprocess_pipeline):
        # list of documents, each document is a list containing words of that document
        self.mr = movie_reviews
        self.pipeline = preprocess_pipeline
        # Corpus as list of documents. Documents as list of sentences. Sentences as list of tokens
        self.unprocessed_corpus, self.labels = self._get_corpus()
        # Corpus as list of documents. Documents as list of tokens
        self.flattened_corpus = self._flatten()
        # Flattened and preprocessed corpus
        self.processed_corpus = self._preprocess()

        self.corpus_words = self.get_corpus_words()
        self.vocab = self._create_vocab()



    def _list_to_str(self, doc) -> str:
        """
        Put all elements of the list into a single string, separating each element with a space.
        """
        return " ".join([w for sent in doc for w in sent])

    def _preprocess(self):
        return self.pipeline(self.flattened_corpus)

    def _flatten(self):
        """
        Returns
        -------
        list[list[str]]
            Each inner list represents a document. Each document is a list of tokens.
        """

        # 3 nested list: each list contain a document, each inner list contains a phrase (until fullstop), each phrase contains words.

        corpus = [[w for w in self._list_to_str(d).split(" ")] for d in self.unprocessed_corpus]
        return corpus

    def _get_corpus(self):
        neg = self.mr.paras(categories = "neg")
        pos = self.mr.paras(categories = "pos")
        labels = [0] * len(pos) + [1] * len(neg)
        return neg + pos, labels

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

        return self.flattened_corpus, self.labels

    def get_sentence_ds(self):
        neg = self.mr.paras(categories = "neg")
        pos = self.mr.paras(categories = "pos")

        pos = [phrase for doc in pos for phrase in doc]
        neg = [phrase for doc in neg for phrase in doc]

        labels = np.array([0] * len(pos) + [1] * len(neg))
        corpus = neg+pos
        return corpus, labels


    def get_corpus_words(self) -> list:
        return [w for doc in self.processed_corpus for w in doc]
    
    def get_embedding_matrix(self, embedding, embedding_dim):
        """
        Returns
        -------
        np.ndarray
            A 2D which each row has the corresponding embedding from the vocabulary
        """
        matrix_length = len(self.vocab)
        embedding_matrix = np.zeros((matrix_length, embedding_dim))
        # If I use torch.zeros directly it crashes (don't know why)
        embedding_matrix = torch.from_numpy(embedding_matrix.copy())
        null_embedding = torch.tensor([0.0]*embedding_dim)
        for idx, key in enumerate(self.vocab.keys()):
            if torch.equal(embedding[key], null_embedding):
                embedding_matrix[idx] = torch.randn(embedding_dim)
            else:
                embedding_matrix[idx] = embedding[key]
                
        return embedding_matrix
    
    def get_indexed_corpus(self):
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
        
        indexed_corpus = [[torch.tensor(vocab[w], dtype=torch.int32) for w in doc] for doc in self.processed_corpus]
        return indexed_corpus, self.labels


    def _create_vocab(self):
        vocab = dict()
        for word in self.corpus_words:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1
        return vocab



    def __len__(self):
        return len(self.flattened_corpus)



if __name__=="__main__":
    from preprocess import MRPipelineTokens
    from torchtext.vocab import GloVe
    global_vectors = GloVe(name='840B', dim=300)
    pipeline = MRPipelineTokens()
    corpus = MovieReviewsCorpus(pipeline)

    embedding_matrix = corpus.get_embedding_matrix(global_vectors, 300)
    print(embedding_matrix)
