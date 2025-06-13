# Meet Robo: your educational chatbot friend

# import necessary libraries
import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # list of sentences
word_tokens = nltk.word_tokenize(raw)  # list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching Lists

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

GK_INPUTS = (
    "tell me a fact", "give me some knowledge", "general knowledge",
    "tell me something interesting", "what's a fun fact?",
    "do you know any trivia?", "any random fact?", "impress me",
    "tell me something new", "i want to learn something", "educate me",
    "fun fact please", "share a fact", "random gk", "knowledge time"
)
GK_RESPONSES = [
    "Did you know? The Eiffel Tower can be 15 cm taller during summer due to thermal expansion.",
    "A group of flamingos is called a 'flamboyance'!",
    "Bananas are berries, but strawberries aren't.",
    "Octopuses have three hearts and blue blood.",
    "The human brain uses about 20% of the body's energy.",
    "Sharks existed before trees were on Earth.",
    "Honey never spoils. Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible!",
    "A bolt of lightning is five times hotter than the surface of the sun.",
    "Wombat poop is cube-shaped!",
    "The Great Wall of China is not visible from space with the naked eye — that's a myth."
]

AI_INPUTS = (
    "tell me about ai", "what is artificial intelligence", "ai trivia", "fact about machine learning",
    "who invented ai", "teach me ai", "deep learning fact"
)
AI_RESPONSES = [
    "Artificial Intelligence aims to make machines mimic human intelligence.",
    "The term 'Artificial Intelligence' was coined in 1956 by John McCarthy.",
    "Machine Learning allows computers to learn from data without being explicitly programmed.",
    "Neural networks are inspired by the human brain and are the foundation of deep learning.",
    "AI powers applications like Siri, Google Assistant, and ChatGPT!",
    "In 1997, IBM’s Deep Blue beat world chess champion Garry Kasparov."
]

NETWORK_INPUTS = (
    "networking fact", "what is the internet", "teach me networking", "fact about IP",
    "how wifi works", "computer network trivia"
)
NETWORK_RESPONSES = [
    "The Internet is a massive network of networks that connects billions of devices worldwide.",
    "IP addresses are like digital addresses — every device connected to the internet has one.",
    "Wi-Fi uses radio waves to transmit data wirelessly.",
    "TCP/IP is the communication protocol that the internet runs on.",
    "The first email was sent over ARPANET, the predecessor of the modern internet.",
    "A router connects different networks and directs traffic between them."
]

PROGRAMMING_INPUTS = (
    "tell me a programming fact", "give me a coding trivia", "programming knowledge", "teach me programming",
    "fact about python", "fact about java", "how did programming start", "tell me about code"
)
PROGRAMMING_RESPONSES = [
    "Python was created by Guido van Rossum and released in 1991.",
    "Java was originally developed by James Gosling at Sun Microsystems in 1995.",
    "The first high-level programming language was Fortran, developed in the 1950s.",
    "C, developed by Dennis Ritchie, was used to build the Unix operating system.",
    "Whitespace is actually a programming language — its syntax is only spaces and tabs!",
    "Programming is the art of telling computers what to do using logic and language."
]

CYBER_INPUTS = (
    "teach me cybersecurity", "fact about hacking", "what is phishing", "internet security tips",
    "cyber trivia", "fact about cyber attacks"
)
CYBER_RESPONSES = [
    "Phishing is a cyber attack where attackers trick you into revealing personal information.",
    "Cybersecurity involves protecting computers, servers, and networks from digital attacks.",
    "Two-factor authentication adds an extra layer of security to your accounts.",
    "A firewall monitors and controls incoming and outgoing network traffic.",
    "The most common password in the world is still '123456'. Change it!",
    "Ethical hackers help organizations find and fix security flaws."
]

OS_INPUTS = (
    "os trivia", "fact about windows", "what is linux", "teach me about operating systems",
    "role of os", "who invented unix"
)
OS_RESPONSES = [
    "An Operating System is software that manages hardware and software resources.",
    "Linux is open-source and powers most web servers in the world.",
    "Windows was first released in 1985 by Microsoft.",
    "Unix, created in the 1970s at Bell Labs, is the foundation for many modern OSes.",
    "An OS handles memory, processes, files, and device management.",
    "macOS, Windows, and Linux are the three most common desktop OSes."
]

DSA_INPUTS = (
    "fact about data structures", "what is an algorithm", "teach me sorting", "cs fundamentals",
    "binary tree trivia", "stack vs queue", "why are algorithms important"
)
DSA_RESPONSES = [
    "Data structures organize and store data efficiently for fast access and modification.",
    "Stacks work on Last In First Out (LIFO), while Queues use First In First Out (FIFO).",
    "Sorting algorithms like QuickSort and MergeSort help organize data in a logical order.",
    "A binary tree has each node with up to two children — left and right.",
    "Algorithms are step-by-step procedures used to solve problems efficiently.",
    "Graphs are used to represent networks like maps or social connections."
]

# Input matcher
def respond_to_input(user_input):
    user_input = user_input.lower()
    if user_input in GREETING_INPUTS:
        return random.choice(GREETING_RESPONSES)
    elif user_input in GK_INPUTS:
        return random.choice(GK_RESPONSES)
    elif user_input in AI_INPUTS:
        return random.choice(AI_RESPONSES)
    elif user_input in NETWORK_INPUTS:
        return random.choice(NETWORK_RESPONSES)
    elif user_input in PROGRAMMING_INPUTS:
        return random.choice(PROGRAMMING_RESPONSES)
    elif user_input in CYBER_INPUTS:
        return random.choice(CYBER_RESPONSES)
    elif user_input in OS_INPUTS:
        return random.choice(OS_RESPONSES)
    elif user_input in DSA_INPUTS:
        return random.choice(DSA_RESPONSES)
    else:
        return None

# Check for greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# TF-IDF Fallback response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response += "I am sorry! I don't understand you"
    else:
        robo_response += sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

# Chat loop
flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots and Computer Science. If you want to exit, type Bye!")

while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response in ['thanks', 'thank you']:
            flag = False
            print("ROBO: You are welcome..")
        else:
            bot_reply = respond_to_input(user_response)
            if bot_reply:
                print("ROBO: " + bot_reply)
            elif greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                print(response(user_response))
    else:
        flag = False
        print("ROBO: Bye! take care..")
