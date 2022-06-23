from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from transformers import BertTokenizer
import re
import unicodedata
import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer


class TfidfRecommender:

    def __init__(self, news_id, tokenization_method="scibert"):
      
        self.news_id = news_id
        if tokenization_method.lower() not in ["none", "nltk", "bert", "scibert"]:
            raise ValueError(
                'Tokenization method must be one of ["none" | "nltk" | "bert" | "scibert"]'
            )
        self.tokenization_method = tokenization_method.lower()

        # Initialize other variables used in this class
        self.tf = TfidfVectorizer()
        self.tfidf_matrix = dict()
        self.tokens = dict()
        self.stop_words = frozenset()
        self.recommendations = dict()
        self.top_k_recommendations = pd.DataFrame()


    def __clean_text(self, text, for_BERT=False, verbose=False):
      
        try:
            # Normalize unicode
            text_norm = unicodedata.normalize("NFC", text)

            # Remove HTML tags
            clean = re.sub("<.*?>", "", text_norm)

            # Remove new line and tabs
            clean = clean.replace("\n", " ")
            clean = clean.replace("\t", " ")
            clean = clean.replace("\r", " ")
            clean = clean.replace("Â\xa0", "")  # non-breaking space

            # Remove all punctuation and special characters
            clean = re.sub(
                r"([^\s\w]|_)+", "", clean
            )  # noqa W695 invalid escape sequence '\s'

            # If you want to keep some punctuation, see below commented out example
            clean = re.sub(r'([^\s\w\-\_\(\)]|_)+','', clean)

            # Skip further processing if the text will be used in BERT tokenization
            if for_BERT is False:
                # Lower case
                clean = clean.lower()
        except Exception:
            if verbose is True:
                print("Cannot clean non-existent text")
            clean = ""

        return clean

    def clean_dataframe(self, df, cols_to_clean, new_col_name="cleaned_text"):
        
        
        cols_to_clean = [['Title'],['Abstract'],[""]]
        # Collapse the table such that all descriptive text is just in a single column
        df = df.replace(np.nan, "", regex=True)
        df[new_col_name] = df[cols_to_clean].apply(lambda cols: " ".join(cols), axis=1)

        # Check if for BERT tokenization
        if self.tokenization_method in ["bert", "scibert"]:
            for_BERT = True
        else:
            for_BERT = False

        # Clean the text in the dataframe
        df[new_col_name] = df[new_col_name].map(
            lambda x: self.__clean_text(x, for_BERT)
        )

        return df


    def tokenize_text(
        self, df_clean, text_col="cleaned_text", ngram_range=(1, 3), min_df=0
    ):
       
        vectors = df_clean[text_col]

        # If a HuggingFace BERT word tokenization method
        if self.tokenization_method in ["bert", "scibert"]:
            # Set vectorizer
            tf = TfidfVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                min_df=min_df,
                stop_words="english",
            )

            # Get appropriate transformer name
            if self.tokenization_method == "bert":
                bert_method = "bert-base-cased"
            elif self.tokenization_method == "scibert":
                bert_method = "allenai/scibert_scivocab_cased"

            # Load pre-trained model tokenizer (vocabulary)
            tokenizer = BertTokenizer.from_pretrained(bert_method)

            # Loop through each item
            vectors_tokenized = vectors.copy()
            for i in range(0, len(vectors)):
                vectors_tokenized[i] = " ".join(tokenizer.tokenize(vectors[i]))

        elif self.tokenization_method == "nltk":
            # NLTK Stemming
            token_dict = {}  # noqa: F841
            stemmer = PorterStemmer()

            def stem_tokens(tokens, stemmer):
                stemmed = []
                for item in tokens:
                    stemmed.append(stemmer.stem(item))
                return stemmed

            def tokenize(text):
                tokens = nltk.word_tokenize(text)
                stems = stem_tokens(tokens, stemmer)
                return stems

         
            tf = TfidfVectorizer(
                tokenizer=tokenize,
                analyzer="word",
                ngram_range=ngram_range,
                min_df=min_df,
                stop_words="english",
            )
            vectors_tokenized = vectors

        elif self.tokenization_method == "none":
            # No tokenization applied
            tf = TfidfVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                min_df=min_df,
                stop_words="english",
            )
            vectors_tokenized = vectors

        # Save to class variable
        self.tf = tf

        return tf, vectors_tokenized


    def fit(self, tf, vectors_tokenized):
      
        self.tfidf_matrix = tf.fit_transform(vectors_tokenized)


    def get_tokens(self):
       
        try:
            self.tokens = self.tf.vocabulary_
        except Exception:
            self.tokens = "Run .tokenize_text() and .fit_tfidf() first"
        return self.tokens


    def get_stop_words(self):
       
        try:
            self.stop_words = self.tf.get_stop_words()
        except Exception:
            self.stop_words = "Run .tokenize_text() and .fit_tfidf() first"
        return self.stop_words


    def __create_full_recommendation_dictionary(self, df_clean):

        # Similarity measure
        cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        # sorted_idx has the indices that would sort the array.
        sorted_idx = np.argsort(cosine_sim, axis=1)

        data = list(df_clean[self.id_col].values)
        len_df_clean = len(df_clean)

        results = {}
        for idx, row in zip(range(0, len_df_clean), data):
            similar_indices = sorted_idx[idx][: -(len_df_clean + 1) : -1]
            similar_items = [(cosine_sim[idx][i], data[i]) for i in similar_indices]
            results[row] = similar_items[1:]

        # Save to class
        self.recommendations = results

    def __organize_results_as_tabular(self, df_clean, k):
        
        # Initialize new dataframe to hold recommendation output
        item_id = list()
        rec_rank = list()
        rec_score = list()
        rec_item_id = list()

        # For each item
        for _item_id in self.recommendations:
            # Information about the item we are basing recommendations off of
            rec_based_on = tmp_item_id = _item_id

            # Get all scores and IDs for items recommended for this current item
            rec_array = self.recommendations.get(rec_based_on)
            tmp_rec_score = list(map(lambda x: x[0], rec_array))
            tmp_rec_id = list(map(lambda x: x[1], rec_array))

            # Append multiple values at a time to list
            item_id.extend([tmp_item_id] * k)
            rec_rank.extend(list(range(1, k + 1)))
            rec_score.extend(tmp_rec_score[:k])
            rec_item_id.extend(tmp_rec_id[:k])

        # Save the output
        output_dict = {
            self.id_col: item_id,
            "rec_rank": rec_rank,
            "rec_score": rec_score,
            "rec_" + self.id_col: rec_item_id,
        }

        # Convert to dataframe
        self.top_k_recommendations = pd.DataFrame(output_dict)

    def recommend_top_k_items(self, df_clean, k=5):
       
        if k > len(df_clean) - 1:
            raise ValueError(
                "Cannot get more recommendations than there are items. Set k lower."
            )
        self.__create_full_recommendation_dictionary(df_clean)
        self.__organize_results_as_tabular(df_clean, k)

        return self.top_k_recommendations


    def __get_single_item_info(self, metadata, rec_id):
        
        # Return row
        rec_info = metadata.iloc[int(np.where(metadata[self.id_col] == rec_id)[0])]

        return rec_info

    def __make_clickable(self, address):
        

        return '<a href="{0}">{0}</a>'.format(address)

    def get_top_k_recommendations(
        self, metadata, query_id, cols_to_keep=[], verbose=True
    ):
       

        # Create subset of dataframe with just recommendations for the item of interest
        df = self.top_k_recommendations.loc[
            self.top_k_recommendations[self.id_col] == query_id
        ].reset_index()

        # Remove id_col of query item
        df.drop([self.id_col], axis=1, inplace=True)

        # Add metadata for each recommended item (rec_<id_col>)
        metadata_cols = metadata.columns.values
        df[metadata_cols] = df.apply(
            lambda row: self.__get_single_item_info(
                metadata, row["rec_" + self.id_col]
            ),
            axis=1,
        )

        # Remove id col added from metadata (already present from self.top_k_recommendations)
        df.drop([self.id_col], axis=1, inplace=True)

        # Rename columns such that rec_ is no longer appended, for simplicity
        df = df.rename(columns={"rec_rank": "rank", "rec_score": "similarity_score"})

        # Only keep columns of interest
        if len(cols_to_keep) > 0:
            # Insert our recommendation scoring/ranking columns
            cols_to_keep.insert(0, "similarity_score")
            cols_to_keep.insert(0, "rank")
            df = df[cols_to_keep]

        # Make URLs clickable if they exist
        if "url" in list(map(lambda x: x.lower(), metadata_cols)):
            format_ = {"url": self.__make_clickable}
            df = df.head().style.format(format_)

        if verbose:
            df

        return  



