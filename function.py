import re
import spacy
import fasttext
import numpy as np

nlp=spacy.load("en_core_web_sm")


def preprocess(text):
    text=text.lower()

    text=re.sub(r'<.*?>','',text)
    
    doc=nlp(text)

    clean=[]

    for token in doc:
        if  token.is_punct or  token.is_space:
            continue
        
        if token.is_stop and token.text not in ["not", "no", "never"]:
            continue

        clean.append(token.lemma_)
    
    clean_string=" ".join(clean)

    clean_string=clean_string.strip()

    return clean_string


model=fasttext.load_model("embeddingModel.bin")


def makevec(text):
    vector=[]
    doc=nlp(text)
    for token in doc:
        vector.append(model.get_word_vector(token.text))
    
    return np.mean(vector,axis=0)





