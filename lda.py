import gensim


class LDA:
    def __init__(self):
        """
        Initialize LDA Class
        """
        pass

    def tokenize_words(self, inputs):
        '''
        Lowercase, tokenize and de-accent sentences to produce the tokens using simple_preprocess function from gensim.utils.

        Args:
            inputs: Input Data. List of N sentences.      
        Returns:
            output: Tokenized list of sentences. List of N lists of tokens.
        '''

        output = []
        for sentence in inputs:
            tokens = gensim.utils.simple_preprocess(sentence, deacc=True)
            output.append(tokens)
        return output


    def remove_stopwords(self, inputs, stop_words):
        """
        Remove stopwords from tokenized words.

        Args:
            inputs: Tokenized list of sentences. List of N lists of tokens.
            stop_words: List of S stop_words. 

        Returns:
            output: Filtered tokenized list of sentences. List of N lists of tokens with stop words removed.
        """

        output = []
        for sentence_tokens in inputs:
            filtered = [word for word in sentence_tokens if word not in stop_words]
            output.append(filtered)
        return output

    def create_dictionary(self, inputs) -> gensim.corpora.dictionary.Dictionary:
        """
        Create dictionary and term document frequency for the input data using Dictionary class of gensim.corpora.

        Args:
            inputs: Filtered tokenized list of sentences. List of N lists of tokens with stop words removed.

        Returns:
            id2word: Gensim Dictionary of index to word map.
            corpus: Term document frequency for each word. List of N lists of tuples.

        """

        id2word = gensim.corpora.Dictionary(inputs)
        corpus = [id2word.doc2bow(text) for text in inputs]
        return id2word, corpus

    def build_LDAModel(self, id2word, corpus, num_topics=10):
        """
        Build LDA Model using LdaMulticore class of gensim.models.

        Args:
            id2word: Gensim Dictionary of index to word map.
            corpus: Term document frequency for each word. List of N lists of tuples.
            num_topics: Number of topics for modeling (int)

        Returns:
            lda_model: LdaMulticore instance.
        """

        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            passes=10,
            workers=2  # Adjust based on your CPU
        )
        return lda_model
