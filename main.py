#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[62]:


from dataclasses import dataclass

@dataclass
class Config:
  # Data
  raw_data_dir = './Data/ABC'
  


# In[63]:


import os 
def load_data(folder_path):
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.txt'):
                try:
                    with open(entry.path, 'r', encoding='utf-8', errors='replace') as file:
                        content = file.read() 
                        yield {
                            'path': entry.path,
                            'filename': entry.name,
                            'content': content,
                            'size': os.path.getsize(entry.path)
                        }                        
                except Exception as e:
                    print(f"Error reading {entry.name}: {e}")
            elif entry.is_dir():
                yield from load_data(entry.path)


# In[64]:


path = Config.raw_data_dir
data = list(load_data(path))

for content in data:
    print(f"Filename: {content['filename']}")
    print(f"Size: {content['size']} bytes")


# # Counting the occurrences of each character  

# In[65]:


def count_chart(data, title='Character Occurrences'):
    from collections import Counter
    all_text = ''.join([item['content'] for item in data])
    char_counts = Counter(all_text)
    print("Chart Count is\n" , char_counts)
    chars = list(char_counts.keys())
    counts = list(char_counts.values())
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=chars, y=counts)
    plt.title(title)
    plt.xlabel('Characters')
    plt.ylabel('Occurrences')
    plt.xticks(rotation=90)
    plt.show()


# In[66]:


count_chart(data, title='Character Occurrences in Arabic Punctuation Dataset')


# In[67]:


from collections import Counter, defaultdict
import string

class CharacterAnalyzer:
    def __init__(self, text: str):
        self.text = text
        self.char_counts = self._count_all_chars()
        
    def _count_all_chars(self) -> dict:
        """Count all characters in the text"""
        return dict(Counter(self.text))
    
    def get_sorted_counts(self, descending: bool = True) -> list:
        """Get character counts sorted by frequency"""
        return sorted(
            self.char_counts.items(), 
            key=lambda x: x[1], 
            reverse=descending
        )
    
    def get_char_frequency(self, char: str) -> int:
        """Get frequency of a specific character"""
        return self.char_counts.get(char, 0)
    
    def get_total_chars(self) -> int:
        """Get total number of characters"""
        return len(self.text)
    
    def get_most_common(self, n: int = 10) -> list:
        """Get n most common characters"""
        sorted_counts = self.get_sorted_counts(descending=True)
        return sorted_counts[:n]
    
    def get_least_common(self, n: int = 10) -> list:
        """Get n least common characters"""
        sorted_counts = self.get_sorted_counts(descending=False)
        return sorted_counts[:n]
    def get_number_of_unique_chars(self) -> int:
        """Get number of unique characters"""
        return len(self.char_counts)
    def get_char_stats(self) -> dict:
        """Get comprehensive character statistics"""
        total = self.get_total_chars()
        unique = self.get_number_of_unique_chars()
        
        return {
            'total_characters': total,
            'unique_characters': unique,
            'most_common': self.get_most_common(5),
            'least_common': self.get_least_common(5)
        }
    
    def visualize_counts(self, top_n: int = 20, title: str = 'Character Frequency Visualization'):
        """visualization of character frequencies"""        
        counter = Counter(self.text)
        most_common = counter.most_common(top_n)
        chars, counts = zip(*most_common)
                
        plt.figure(figsize=(12, 6))
        sns.barplot(x=chars, y=counts)
        plt.title(title)
        plt.xlabel('Characters')
        plt.ylabel('Occurrences')
        plt.show()
        

# Usage
text = ''.join([item['content'] for item in data])

analyzer = CharacterAnalyzer(text)

print("Least Common Characters:", analyzer.get_least_common(40))
analyzer.visualize_counts(top_n=30)


# ### as we see, there are too much characters that must be removed, so we're going to specify our vocabulary

# In[68]:


# Arabic Alphabet (Modern Standard Arabic)
ARABIC_LETTERS = [
    # Arabic Alphabet
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص',
    'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي',
    
    # Hamza forms and variations
    'أ', 'إ', 'آ', 'ؤ', 'ئ', 'ء',
    
    # Ta Marbuta
    'ة',
    
    # Alif Maqsura
    'ى',
    
    # Lam-Alif 
    'لا', 
    
    # Additional letters used in various Arabic dialects
    'پ', 'چ', 'ژ', 'گ', 'ڤ',  # Used in some dialects/regions
]

# Arabic Numbers (Eastern Arabic numerals)
ARABIC_NUMERALS = [
    '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'  # 0-9
]

# Standard Western Numbers (also commonly used)
WESTERN_NUMBERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Arabic Punctuation and Diacritics
ARABIC_PUNCTUATION = [
    '،',  # Arabic comma
    '؛',  # Arabic semicolon
    '؟',  # Arabic question mark
    '!',  # Exclamation mark 
    '.',  # Full Stop
    ':',  # Colon
]
LATIN_PUNCTUATION = [
    ',', # Latin comma
    ';', # Latin semicolon
    '?', # Latin question mark
 ]

LATIN_TO_ARABIC_PUNCTUATION = {
    ',': '،',
    ';': '؛',
    '?': '؟',
}
LATIN_TO_ARABIC_NUMBERS = {
    '0': '٠',
    '1': '١',
    '2': '٢',
    '3': '٣',
    '4': '٤',
    '5': '٥',
    '6': '٦',
    '7': '٧',
    '8': '٨',
    '9': '٩'    
}
# Arabic Diacritics 
ARABIC_DIACRITICS = [
    # Short vowels
    # 'َ',  # Fatha
    # 'ُ',  # Damma
    # 'ِ',  # Kasra
    
    # # Tanween (nunation)
    # 'ً',  # Fathatan
    # 'ٌ',  # Dammatan
    # 'ٍ',  # Kasratan
    
    # # Sukun
    # 'ْ',  # Sukun
    
    # Shadda
    'ّ',  # Shadda (gemination)
]

# Combine everything
ARABIC_VOCABULARY = (
    ARABIC_LETTERS + 
    ARABIC_NUMERALS + 
    WESTERN_NUMBERS + 
    ARABIC_PUNCTUATION + 
    LATIN_PUNCTUATION + 
    ARABIC_DIACRITICS
)

# Or create separate lists for different purposes
CHART_SETS = {
    'letters': ARABIC_LETTERS,
    'arabic_numerals': ARABIC_NUMERALS,
    'western_numbers': WESTERN_NUMBERS,
    'punctuation': ARABIC_PUNCTUATION,
    'diacritics': ARABIC_DIACRITICS,
    'all': ARABIC_VOCABULARY,
}

# Create a dictionary with character indices
def create_arabic_vocab_dict(include_special_tokens: bool = True) -> dict:
    """
    Create a complete Arabic vocabulary dictionary with indices
    """
    # Base vocabulary
    base_vocab = ARABIC_VOCABULARY.copy()
    
    # Add special tokens if requested
    special_tokens = []
    if include_special_tokens:
        special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[MASK]', '[SEP]', '[CLS]']
        base_vocab = special_tokens + base_vocab
    
    # Create mapping dictionaries
    char_to_idx = {char: idx for idx, char in enumerate(base_vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocabulary': base_vocab,
        'vocab_size': len(base_vocab),
        'special_tokens': special_tokens if include_special_tokens else [],
        'stats': {
            'letters': len(ARABIC_LETTERS),
            'arabic_numerals': len(ARABIC_NUMERALS),
            'western_numbers': len(WESTERN_NUMBERS),
            'punctuation': len(ARABIC_PUNCTUATION),
            'diacritics': len(ARABIC_DIACRITICS),
        }
    }




# In[69]:


def cleanData(text: str) -> str:
    """Clean text by removing unwanted characters"""
    allowed_chars = set(ARABIC_VOCABULARY + list(string.whitespace))
    cleaned_text = ''.join([char for char in text if char in allowed_chars])
    return cleaned_text


# ### Statistics before cleaning:

# In[70]:


print(analyzer.get_char_stats())


# ### Statistics after cleaning

# In[71]:


cleaned_data = cleanData(text)
new_analyzer = CharacterAnalyzer(cleaned_data)
print(new_analyzer.get_char_stats())


# In[72]:


print(len(ARABIC_VOCABULARY) - new_analyzer.get_number_of_unique_chars())


# ### so there are 6 characters that are not presented in the dataset

# In[73]:


def get_not_presented_charters(vocabulary: list, analyzer: CharacterAnalyzer) -> list:
    """Get characters from vocabulary not present in the analyzed text"""
    not_presented = [char for char in vocabulary if analyzer.get_char_frequency(char) == 0]
    return not_presented


# In[74]:


get_not_presented_charters(ARABIC_VOCABULARY, new_analyzer)


# In[75]:


class ArabicPunctuationProcessor:
    """
    Processor for Arabic text for punctuation classification task
    Handles word/character level processing, punctuation extraction and data preparation
    """
    ARABIC_LETTERS = [
    # Arabic Alphabet
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص',
    'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي',
    
    # Hamza forms and variations
    'أ', 'إ', 'آ', 'ؤ', 'ئ', 'ء',
    
    # Ta Marbuta
    'ة',
    
    # Alif Maqsura
    'ى',
    
    # Lam-Alif 
    'لا', 
    
    # Additional letters used in various Arabic dialects
    'پ', 'چ', 'ژ', 'گ', 'ڤ',  # Used in some dialects/regions
    ]

    # Arabic Numbers (Eastern Arabic numerals)
    ARABIC_NUMERALS = [
        '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'  # 0-9
    ]

    # Standard Western Numbers (also commonly used)
    WESTERN_NUMBERS = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]

    # Arabic Punctuation and Diacritics
    ARABIC_PUNCTUATION = [
        '،',  # Arabic comma
        '؛',  # Arabic semicolon
        '؟',  # Arabic question mark
        '!',  # Exclamation mark 
        '.',  # Full Stop
        ':',  # Colon
    ]
    LATIN_PUNCTUATION = [
        ',', # Latin comma
        ';', # Latin semicolon
        '?', # Latin question mark
    ]

    LATIN_TO_ARABIC_PUNCTUATION = {
        ',': '،',
        ';': '؛',
        '?': '؟',
    }
    LATIN_TO_ARABIC_NUMBERS = {
        '0': '٠',
        '1': '١',
        '2': '٢',
        '3': '٣',
        '4': '٤',
        '5': '٥',
        '6': '٦',
        '7': '٧',
        '8': '٨',
        '9': '٩'    
    }
    # Arabic Diacritics 
    ARABIC_DIACRITICS = [
        # # Short vowels
        # 'َ',  # Fatha
        # 'ُ',  # Damma
        # 'ِ',  # Kasra
        
        # # Tanween (nunation)
        # 'ً',  # Fathatan
        # 'ٌ',  # Dammatan
        # 'ٍ',  # Kasratan
        
        # # Sukun
        # 'ْ',  # Sukun
        
        # Shadda
        'ّ',  # Shadda (gemination)
        
        # Madda
        'آ',  # Madda (already in letters, kept for completeness)
        
        # Wasla
        'ٱ',  # Alif Wasla
    ]

    # Combine everything
    ARABIC_CHART_VOCABULARY = (
        ARABIC_LETTERS + 
        ARABIC_NUMERALS + 
        WESTERN_NUMBERS + 
        ARABIC_PUNCTUATION + 
        LATIN_PUNCTUATION +
        ARABIC_DIACRITICS
    )

    PUNCTUATION_CLASSES = {
        0: "none",          # No punctuation
        1: "comma",         # '،' or ','
        2: "semicolon",     # '؛' or ';'
        3: "colon",         # ':'
        4: "fullstop",      # '.'
        5: "question",      # '؟' or '?'
        6: "exclamation"    # '!'
    }
    PUNCTUATION_TO_CLASS = {
        '':  0, 
        '،': 1,
        '؛': 2, 
        ':': 3,
        '.': 4,
        '؟': 5,
        '!': 6
    }
    
    def __init__(self, text: str):
        self.text = text 
        self.cleaned_text = self._normalize_text(self._clean_text())
    
    # To clean the text from non-arabic charts
    def _clean_text(self) -> str:
        """Clean text by removing unwanted characters"""
        allowed_chars = set(self.ARABIC_CHART_VOCABULARY + list(string.whitespace))
        cleaned_text = ''.join([char for char in self.text if char in allowed_chars])
        return cleaned_text
    
    # To normalize the text by converting Latin punctuation and numbers to Arabic equivalents
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting Latin punctuation and numbers to Arabic equivalents"""
        norm_text = ''.join([
            self.LATIN_TO_ARABIC_PUNCTUATION.get(char, 
            self.LATIN_TO_ARABIC_NUMBERS.get(char, char)) 
            for char in text
        ])
        return norm_text 
    
    def _is_punctuation(self, char) :
        return char in self.PUNCTUATION_TO_CLASS.keys()
    
    def tokenize_arabic_words(self) -> list:
        """Tokenize cleaned Arabic text into words"""
        words = self.cleaned_text.split()
        return words
    def split_words_labels(self) -> str :
        "Split text into words and labels for punctuation classification"
        
        words = [] 
        labels = [] 

        for word in self.tokenize_arabic_words():
            if self._is_punctuation(word[0]):
                if not words:
                    continue 
                labels[-1] = self.PUNCTUATION_TO_CLASS[word[0]]
            elif self._is_punctuation(word[-1]):
                new_word = ""
                # handling the case if the word ends with multiple punctuations
                for char in word:
                    if char != word[-1]:
                        new_word += char
                        
                words.append(new_word)
                labels.append(self.PUNCTUATION_TO_CLASS[word[-1]])
                
            
            else:
                words.append(word)
                labels.append(self.PUNCTUATION_TO_CLASS[''])
        
        return words, labels 
    
    
        
        
         
        


# In[76]:


text = "مرحبًا، كيف حالك؟ 123.1 ???؟؟؟ ,,,,,,,,,,,،،،،/ ـ،/:"""""""

text2 = """
قَالَ الطَّبِيبُ: هَلْ تُعَانِي مِنْ صُدَاعٍ مُسْتَمِرٍّ، أَمْ أَنَّ الأَلَمَ يَظْهَرُ أَحْيَانًا فَقَطْ؟
أَجَابَهُ الْمَرِيضُ: أَشْعُرُ بِالتَّعَبِ مُنْذُ ثَلَاثَةِ أَيَّامٍ، وَلَا أَسْتَطِيعُ النَّوْمَ جَيِّدًا!
وَأَضَافَ: هَلْ هٰذَا طَبِيعِيٌّ؟! أَمْ يَجِبُ أَنْ أُرَاجِعَ الْمُسْتَشْفَى فَوْرًا؟
فِي الْحَقِيقَةِ، قَالَ الطَّبِيبُ؛ إِنَّ هٰذِهِ الأَعْرَاضَ شَائِعَةٌ، لَكِنَّهَا قَدْ تُشِيرُ إِلَى مُشْكِلَةٍ خَطِيرَةٍ.
تَنَاوَلِ الدَّوَاءَ بِانْتِظَامٍ، وَاشْرَبِ الْمَاءَ، وَلَا تُهْمِلِ الرَّاحَةَ... ثُمَّ رَاقِبْ حَالَتَكَ.
"""
processor = ArabicPunctuationProcessor(text2)

print("Original Text:", processor.text)
print("Cleaned Text:", processor.cleaned_text)
print("Tokenized Words:", processor.tokenize_arabic_words())

words, labels = processor.split_words_labels()

print("is ، punctuation:", processor._is_punctuation('،'))


print("words are" , words) 
print("labels are" , labels)

for idx, word in enumerate(words):
    print(f"Word: {word} , class : {labels[idx]} , punctuation : {list(processor.PUNCTUATION_TO_CLASS.keys())[labels[idx]]}")


# In[ ]:




