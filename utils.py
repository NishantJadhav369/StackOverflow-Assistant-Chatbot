import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    import numpy as np
    embeddings = {}
    with open(embeddings_path,'r') as fp:
        for i in fp:
            temp = i.split('\t')
            embeddings[temp[0]] = np.array(temp[1:]).astype(np.float32)
    embed_dim = len(embeddings[list(embeddings.keys())[0]])
    return embeddings,embed_dim
    ########################

    # remove this when you're done
    raise NotImplementedError(
        "Open utils.py and fill with your code. In case of Google Colab, download"
        "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
        "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    vocab = embeddings.keys()
    
    if question=='':
        return np.zeros(dim)

    sentence_embed = []
    count_known_words= 0
    for word in question.split():
        if word in vocab:
            word_embed_array = embeddings[word]
            sentence_embed.append(word_embed_array)
            count_known_words += 1
    
    if count_known_words ==0:
        return np.zeros(dim)
    
    sentence_embed =  np.array(sentence_embed).astype(np.float)
    
    sum_embed =np.sum(sentence_embed,axis=0)
    #print(sum_embed)
    #avg_embed = [float(x) / count_known_words for x in sum_embed]
    avg_embed = sum_embed/count_known_words
    #avg_embed = sum_embed/count_known_words
    
    return avg_embed
    ########################

    # remove this when you're done
    raise NotImplementedError(
        "Open utils.py and fill with your code. In case of Google Colab, download"
        "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
        "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
