from django.db import models
import nltk
from nltk.stem import PorterStemmer
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet')

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(10000, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)
        
    def forward(self, x):
        # x = self.linear1(x)
        x = torch.flatten(x, 0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def classify_spam(message):
    model = LogisticRegression()
    model.load_state_dict(torch.load('W:/SpamOrHamProject/SpamOrHamBack/api/AIModel/SpamClassification.pth'))
    model.eval()
    
    cv = CountVectorizer(max_features=10000, stop_words='english')
    # processed_message = preprocess_message(message)
    processed_message = []
    for n in range(10000):
        processed_message.append(message)

    cv.fit(processed_message)
    vectorized_message = cv.transform(processed_message).toarray()
    
    with torch.no_grad():
        tensor_message = Variable(torch.from_numpy(vectorized_message)).float()
        output = model(tensor_message)
        _, predicted_label = torch.max(output, 0)
    
    return 'Spam' if predicted_label.item() == 1 else 'Not Spam'

def preprocess_message(message):
    remove_non_alphabets = lambda x: re.sub(r'[^a-zA-Z]', ' ', x)
    tokenize = lambda x: word_tokenize(x)
    ps = PorterStemmer()
    stem = lambda w: [ps.stem(x) for x in w]
    lemmatizer = WordNetLemmatizer()
    leammtizer = lambda x: [lemmatizer.lemmatize(word) for word in x]

    processed_message = remove_non_alphabets(message)
    processed_message = tokenize(processed_message)
    processed_message = stem(processed_message)
    processed_message = leammtizer(processed_message)
    processed_message = ' '.join(processed_message)

    return processed_message