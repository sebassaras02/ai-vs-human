from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.stem import PorterStemmer
import re


class TextProcessing:
    """
    This class contains all methods to process text data.
    """
    def __init__(self, language : str = 'english'):
        self.list_stopwords = list(set(stopwords.words(language)))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def tokenize(self, text : str) -> list:
        """
        This function takes a string and returns a list of words in the string.

        Args:
            text : A string of words
        
        Returns:
            the tokens
        """
        return text.split()
    
    def remove_stopwords(self, list_tokens : list) -> list:
        """
        This function removes the stopwords from the list of tokens.

        Args: 
            list_tokens : list of tokens to process
        
        Returns:
            list of tokens with the stopwords removed
        """
        return [word for word in list_tokens if word not in self.list_stopwords]

    def lemmatize_tokens(self, list_tokens : list) -> list:
        """
        This function lemmatizes a list of tokens.

        Args:
            list_tokens : list of tokens
            lemmatizer : instance of WordNetLemmatizer
        
        Returns:
            list of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in list_tokens]
    
    def steem_tokens(self, list_tokens : list) -> list:
        """
        This function steems a list of tokens.

        Args:
            list_tokens : list of tokens
        
        Returns:
            list of steemed tokens
        """
        return [self.stemmer.stem(word) for word in list_tokens]
        
    
    def lowercase_tokens(self, list_tokens : list) -> list:
        """"
        This function receives a list of tokens and returns a list of tokens in lowercase

        Args:
            list_tokens: list of strings

        Returns:
            list of strings
        """
        return [word.lower() for word in list_tokens]
    
    def remove_short_tokens(self, token_list : list, min_length : int = 3) -> list:
        """
        This function removes words from a list of tokens that are shorter than min_length.

        Args:
            token_list: list of strings
            min_length: int, minimum length of the words to keep
        
        Returns:
            list of strings 
        """
        return [word for word in token_list if len(word) >= min_length]
    
    def remove_punctuation(self, text : str) -> str:
        """
        This function removes punctuation from a list of tokens.

        Args:
            token_list: list of strings
        
        Returns:
            list of strings
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\d', '', text)
        return text

    def join_tokens_cleaned(self, token_list : list ) -> list:
        """
        This function joins the tokens in a list

        Args:
            token_list : list of tokens cleaned
        
        Returns:
            text : final phrase
        """
        return " ".join(token_list)

    def fit_transform(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        This function receives a dataframe and applies the text processing methods to the text column.

        Args:
            df : pandas DataFrame with a column named 'text'

        Returns:
            df : pandas DataFrame with a column named 'processed_text'
        """
        df['text'] = df['text'].apply(lambda x: self.remove_punctuation(x))
        df['processed_text'] = df['text'].apply(lambda x: self.tokenize(x))
        df['processed_text'] = df['processed_text'].apply(lambda x: self.lowercase_tokens(x))
        df['processed_text'] = df['processed_text'].apply(lambda x: self.remove_stopwords(x))
        df['processed_text'] = df['processed_text'].apply(lambda x: self.remove_short_tokens(x))
        df['processed_text'] = df['processed_text'].apply(lambda x: self.steem_tokens(x))
        df['processed_text'] = df['processed_text'].apply(lambda x: self.join_tokens_cleaned(x))

        return df
        

