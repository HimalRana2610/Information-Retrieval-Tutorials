"""
Tutorial 2: Basic Text Processing Techniques in NLP
=====================================================
Objective: Understand and implement tokenization, stemming,
           lemmatization, and stop-word removal using NLTK.

Input: "The students are studying different subjects and they are
        enjoying their studies. The teachers are teaching the subjects
        very effectively."
"""

import nltk

# ─── Download required NLTK data ────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# =====================================================================
#  Input Text
# =====================================================================
text = ("The students are studying different subjects and they are "
        "enjoying their studies. The teachers are teaching the "
        "subjects very effectively.")

print("=" * 65)
print(" Tutorial 2: Basic Text Processing Techniques in NLP")
print("=" * 65)

print("\n--- Original Text ---")
print(text)

# =====================================================================
#  1. Tokenization
# =====================================================================
tokens = word_tokenize(text)

print("\n--- Step 1: Tokenization ---")
print("Tokens:", tokens)
print(f"Total tokens: {len(tokens)}")

# =====================================================================
#  2. Stop Words Removal
# =====================================================================
stop_words = set(stopwords.words('english'))

# Show which stop words were found in the text
found_stop_words = [t for t in tokens if t.lower() in stop_words]
filtered_tokens = [t for t in tokens if t.lower() not in stop_words
                   and t.isalpha()]

print("\n--- Step 2: Stop Words Removal ---")
print(f"Stop words identified: {found_stop_words}")
print(f"Text after removing stop words: {filtered_tokens}")
print(f"Tokens remaining: {len(filtered_tokens)}")

# =====================================================================
#  3. Stemming (Porter Stemmer)
# =====================================================================
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\n--- Step 3: Stemming (Porter Stemmer) ---")
print(f"{'Original':<20} {'Stemmed':<20}")
print("-" * 40)
for original, stemmed in zip(filtered_tokens, stemmed_words):
    print(f"{original:<20} {stemmed:<20}")

# =====================================================================
#  4. Lemmatization (WordNet Lemmatizer)
# =====================================================================
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word.lower(), pos='v')
                    for word in filtered_tokens]

print("\n--- Step 4: Lemmatization (WordNet Lemmatizer) ---")
print(f"{'Original':<20} {'Lemmatized':<20}")
print("-" * 40)
for original, lemmatized in zip(filtered_tokens, lemmatized_words):
    print(f"{original:<20} {lemmatized:<20}")

# =====================================================================
#  5. Comparison: Stemming vs Lemmatization
# =====================================================================
print("\n--- Step 5: Comparison of Stemming and Lemmatization ---")
print(f"{'Original':<20} {'Stemmed':<20} {'Lemmatized':<20}")
print("-" * 60)
for orig, stem, lemma in zip(filtered_tokens, stemmed_words,
                             lemmatized_words):
    # Highlight differences
    marker = " <-- different" if stem != lemma else ""
    print(f"{orig:<20} {stem:<20} {lemma:<20}{marker}")

# =====================================================================
#  Answers to Conceptual Questions
# =====================================================================
print("\n" + "=" * 65)
print(" Answers to Conceptual Questions")
print("=" * 65)

print("""
Q1: What is the difference between stemming and lemmatization?

   Stemming:       Chops off word endings using simple rules to get
                   a root form (stem). May produce non-dictionary
                   words (e.g., 'studying' -> 'studi').

   Lemmatization:  Uses vocabulary and morphological analysis to
                   return the meaningful dictionary form (lemma)
                   (e.g., 'studying' -> 'study').

Q2: Which technique produces more meaningful English words?

   Lemmatization produces more meaningful words because it returns
   actual dictionary entries, while stemming can produce truncated
   forms that are not real words.

Q3: Why are stop words removed during NLP preprocessing?

   Stop words (the, is, are, and, etc.) are extremely common words
   that carry little semantic meaning. Removing them reduces noise,
   decreases the dimensionality of the data, and allows NLP models
   to focus on the content-bearing words.

Q4: What is the purpose of tokenization?

   Tokenization breaks a continuous text string into discrete units
   (tokens/words) so that each unit can be individually analyzed,
   counted, or processed by NLP algorithms.

Q5: Two real-world applications of these text-processing techniques:

   1. Search Engines: Tokenization, stemming, and stop-word removal
      help index and match user queries to relevant documents
      (e.g., searching "studying" also finds "studies" and "study").

   2. Sentiment Analysis: Cleaning customer reviews with these
      techniques allows ML models to accurately classify feedback
      as positive or negative.
""")

# =====================================================================
#  Conclusion
# =====================================================================
print("=" * 65)
print(" Conclusion")
print("=" * 65)
print("""
Text processing is an important first step in NLP.
- Tokenization divides text into smaller units.
- Stop-word removal eliminates frequently occurring words with
  limited analytical value.
- Stemming reduces words to their stems.
- Lemmatization converts words into meaningful dictionary forms.

These techniques help prepare textual data for applications such as
sentiment analysis, search engines, text classification, and chatbots.
""")