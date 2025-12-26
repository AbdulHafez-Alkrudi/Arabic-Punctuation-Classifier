# %% [markdown]
# # 🔧 Data Preprocessing Pipeline for Arabic Punctuation Dataset
# ## SSAC-UNPC Component Preprocessing
# 
# ---
# 
# ### 📋 Table of Contents
# 
# 1. [Introduction & Setup](#1-introduction--setup)
# 2. [Part 1: Problem Inspection (Before Preprocessing)](#2-part-1-problem-inspection-before-preprocessing)
#    - 2.1 Character-Level Issues
#    - 2.2 Punctuation Issues
#    - 2.3 Sentence-Level Issues
#    - 2.4 Special Pattern Issues
# 3. [Part 2: Mandatory Preprocessing Steps](#3-part-2-mandatory-preprocessing-steps)
#    - 3.1 Remove Diacritics (Tashkeel)
#    - 3.2 Normalize Alef Variations
#    - 3.3 Normalize Teh Marbuta and Alef Maksura
#    - 3.4 Remove Out-of-Vocabulary Characters
#    - 3.5 Remove Latin Letters
#    - 3.6 Unify Numbers (Arabic Numerals)
#    - 3.7 Unify Punctuation (Arabic Punctuation)
#    - 3.8 Handle Consecutive Punctuation
#    - 3.9 Normalize Whitespace and Punctuation Spacing
#    - 3.10 Remove Empty and Very Short Lines
#    - 3.11 Process Long Sentences
# 4. [Part 3: Optional Preprocessing Steps (For Experimentation)](#4-part-3-optional-preprocessing-steps)
#    - 4.1 Separate Waw Conjunction from Words
#    - 4.2 Stopword Handling Strategies
#    - 4.3 Number Token Replacement
#    - 4.4 Rare Word Handling
#    - 4.5 Sentence Length Normalization
#    - 4.6 Remove/Replace Foreign Terms
# 5. [Part 4: Complete Preprocessing Pipeline](#5-part-4-complete-preprocessing-pipeline)
# 6. [Part 5: Post-Preprocessing Inspection](#6-part-5-post-preprocessing-inspection)
# 7. [Part 6: Save Preprocessed Data](#7-part-6-save-preprocessed-data)
# 8. [Summary & Recommendations](#8-summary--recommendations)
# 
# ---

# %% [markdown]
# ## 1. Introduction & Setup
# 
# ### 🎯 Purpose of This Notebook
# 
# This notebook implements a comprehensive preprocessing pipeline for the SSAC-UNPC 
# component of the Arabic Punctuation Dataset. Based on the EDA findings, we address:
# 
# **Issues Identified in EDA:**
# 
# | Issue | Severity | Solution |
# |-------|----------|----------|
# | Diacritics present (0.27%) | Medium | Remove all tashkeel |
# | Mixed Arabic/Latin punctuation | High | Normalize to Arabic |
# | Mixed Arabic/Western numerals | Medium | Normalize to Arabic |
# | Multiple Alef forms | Low | Normalize to bare Alef |
# | Out-of-vocabulary characters | Low | Remove or replace |
# | Latin letters in text | Medium | Remove |
# | Very short sentences (<3 words) | Medium | Filter out |
# | Very long sentences (>100 words) | Low | Truncate or split |
# | Consecutive punctuation | Low | Handle appropriately |
# | Attached punctuation | Medium | Add spacing |
# 
# ### 📊 Expected Outcomes
# 
# After preprocessing:
# - Clean, consistent Arabic text
# - Unified punctuation system (Arabic only)
# - Unified numeral system (Arabic only)
# - No diacritics or special marks
# - Proper spacing around punctuation
# - Filtered short/problematic sentences
# - Ready for tokenization and model training

# %%
# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

# -----------------------------
# Standard Library Imports
# -----------------------------
import os
import re
import sys
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional, Generator, Callable
from dataclasses import dataclass, field

# -----------------------------
# Data Analysis Libraries
# -----------------------------
import numpy as np

# -----------------------------
# Progress Bar
# -----------------------------
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed. Install with: pip install tqdm")

# %%
# ============================================================================
# SECTION 2: LOGGER CONFIGURATION
# ============================================================================

class NotebookLogger:
    """
    Minimal unified logger for Jupyter notebooks.
    
    - Prints to notebook output
    - Appends logs to a file
    - No timestamps
    - No session headers
    """

    def __init__(
        self,
        log_file: str | Path = "preprocessing.log",
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        self.log_file = Path(log_file)
        self.enable_console = enable_console
        self.enable_file = enable_file

        if self.enable_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, message: str):
        if self.enable_console:
            print(message, end="")

        if self.enable_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(message)

    def info(self, message: str):
        self._write(f"{message}\n")

    def warn(self, message: str):
        self._write(f"⚠️  WARNING: {message}\n")

    def error(self, message: str):
        self._write(f"❌ ERROR: {message}\n")

    def success(self, message: str):
        self._write(f"✅ {message}\n")

    def section(self, title: str):
        block = (
            "\n" + "=" * 70 +
            f"\n{title}\n" +
            "=" * 70 + "\n"
        )
        self._write(block)

    def subsection(self, title: str):
        self._write(f"\n--- {title} ---\n")


# Initialize logger
logger = NotebookLogger(log_file="logs/preprocessing.log")

# %%
# ============================================================================
# SECTION 3: CONFIGURATION
# ============================================================================

@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing pipeline.
    
    Separates mandatory and optional preprocessing steps.
    """
    
    # -----------------------------
    # File Paths
    # -----------------------------
    input_dir: str = "../SSAC-UNPC"
    output_dir: str = "preprocessed_data"
    log_dir: str = "logs"
    
    # -----------------------------
    # Mandatory Preprocessing (Always Applied)
    # -----------------------------
    remove_diacritics: bool = True
    normalize_alef: bool = True
    normalize_teh_marbuta: bool = True  # ة → ه (optional, some keep it)
    normalize_alef_maksura: bool = True  # ى → ي
    remove_tatweel: bool = True
    remove_latin_letters: bool = True
    remove_oov_chars: bool = True
    unify_numbers_to_arabic: bool = True
    unify_punctuation_to_arabic: bool = True
    handle_consecutive_punct: bool = True
    normalize_whitespace: bool = True
    add_punct_spacing: bool = True
    
    # -----------------------------
    # Sentence Filtering
    # -----------------------------
    min_words: int = 3
    max_words: int = 100
    remove_empty_lines: bool = True
    
    # -----------------------------
    # Optional Preprocessing (For Experimentation)
    # -----------------------------
    separate_waw_conjunction: bool = False
    remove_stopwords: bool = False
    replace_numbers_with_token: bool = False
    replace_rare_words: bool = False
    rare_word_threshold: int = 5
    remove_foreign_terms: bool = False
    
    # -----------------------------
    # Processing Parameters
    # -----------------------------
    sample_size: Optional[int] = None  # None = process all
    chunk_size: int = 100000  # Lines per chunk for memory efficiency
    random_seed: int = 42


# Initialize configuration
config = PreprocessingConfig()

# Create output directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

logger.info("✅ Configuration initialized!")
logger.info(f"   Input directory: {config.input_dir}")
logger.info(f"   Output directory: {config.output_dir}")

# %%
# ============================================================================
# SECTION 4: ARABIC CHARACTER DEFINITIONS
# ============================================================================

# -----------------------------
# Valid Arabic Characters
# -----------------------------
# Based on EDA findings: All Arabic letters found in dataset

ARABIC_LETTERS = set(
    'ء آ أ ؤ إ ئ ا ب ة ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي ى'
    .split()
)

# Extended set including less common letters (Persian/Urdu influence in names)
ARABIC_LETTERS_EXTENDED = ARABIC_LETTERS | set('پ چ ژ گ ڤ')

# -----------------------------
# Arabic Diacritics (Tashkeel)
# -----------------------------
ARABIC_DIACRITICS = {
    '\u064B': 'Fathatan',   # ً
    '\u064C': 'Dammatan',   # ٌ
    '\u064D': 'Kasratan',   # ٍ
    '\u064E': 'Fatha',      # َ
    '\u064F': 'Damma',      # ُ
    '\u0650': 'Kasra',      # ِ
    '\u0651': 'Shadda',     # ّ
    '\u0652': 'Sukun',      # ْ
}

DIACRITICS_PATTERN = re.compile(r'[\u064B-\u0652]')

# -----------------------------
# Arabic Numerals
# -----------------------------
ARABIC_NUMERALS = '٠١٢٣٤٥٦٧٨٩'
WESTERN_NUMERALS = '0123456789'

# Mapping tables
WESTERN_TO_ARABIC_NUMS = str.maketrans(WESTERN_NUMERALS, ARABIC_NUMERALS)
ARABIC_TO_WESTERN_NUMS = str.maketrans(ARABIC_NUMERALS, WESTERN_NUMERALS)

# -----------------------------
# Punctuation Marks
# -----------------------------
# Target punctuation (Arabic)
ARABIC_PUNCTUATION = {
    '،': 'Arabic Comma',
    '؛': 'Arabic Semicolon',
    '؟': 'Arabic Question Mark',
    '.': 'Full Stop',
    ':': 'Colon',
    '!': 'Exclamation Mark',
}

# Latin equivalents to normalize
LATIN_TO_ARABIC_PUNCT = {
    ',': '،',   # Latin comma → Arabic comma
    ';': '؛',   # Latin semicolon → Arabic semicolon
    '?': '؟',   # Latin question mark → Arabic question mark
}

# All valid punctuation marks (after normalization)
VALID_PUNCTUATION = set(ARABIC_PUNCTUATION.keys())

# Sentence terminal marks
SENTENCE_TERMINALS = {'.', '؟', '!'}

# -----------------------------
# Alef Variations
# -----------------------------
ALEF_VARIATIONS = {
    'أ': 'ا',  # Alef with Hamza Above
    'إ': 'ا',  # Alef with Hamza Below
    'آ': 'ا',  # Alef with Madda
    'ٱ': 'ا',  # Alef Wasla
}

# -----------------------------
# Other Normalizations
# -----------------------------
# Teh Marbuta (ة) - some normalize to ه, others keep it
# Alef Maksura (ى) - normalize to ي

# -----------------------------
# Arabic Stopwords
# -----------------------------
ARABIC_STOPWORDS = set([
    # Prepositions
    'في', 'من', 'على', 'إلى', 'الى', 'عن', 'مع', 'بين', 'عند', 'حتى', 'منذ',
    'الي', 'فى', 'علي',
    # Demonstratives
    'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك',
    # Relative pronouns
    'التي', 'الذي', 'اللذان', 'اللتان', 'الذين', 'اللاتي', 'اللواتي',
    # Conjunctions
    'و', 'أو', 'او', 'ثم', 'لكن', 'بل', 'إذا', 'لو', 'إذ', 'ف',
    # Particles
    'أن', 'ان', 'إن', 'قد', 'لا', 'ما', 'لم', 'لن', 'ل', 'ب', 'ك',
    # Pronouns
    'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم', 'انا', 'انت',
    # Auxiliary verbs
    'كان', 'كانت', 'يكون', 'تكون', 'كانوا', 'ليس', 'ليست',
    # Others
    'كل', 'بعض', 'أي', 'اي', 'غير', 'بعد', 'قبل', 'حيث', 'عندما',
    'حول', 'دون', 'ضد', 'خلال', 'عبر', 'نحو', 'فوق', 'تحت',
    # Common function words
    'وفي', 'ومن', 'وعلى', 'والى', 'ومع', 'وهو', 'وهي', 'وهذا', 'وهذه',
    'فإن', 'فان', 'وإن', 'وان', 'لأن', 'لان', 'كما', 'مما', 'إنه', 'انه',
    'إنها', 'انها', 'أنه', 'أنها', 'ذات', 'لها', 'له', 'لهم', 'بها', 'به',
    'فيها', 'فيه', 'منها', 'منه', 'عنها', 'عنه', 'إليها', 'إليه',
    'عليها', 'عليه', 'معها', 'معه', 'بينها', 'بينهم',
])

# -----------------------------
# Waw Conjunction Patterns
# -----------------------------
# Words that commonly start with و (waw) as conjunction
WAW_CONJUNCTION_MIN_LENGTH = 3  # Only separate if remaining word has 3+ chars

logger.info("✅ Arabic character definitions loaded!")
logger.info(f"   - Arabic letters: {len(ARABIC_LETTERS)}")
logger.info(f"   - Diacritics: {len(ARABIC_DIACRITICS)}")
logger.info(f"   - Stopwords: {len(ARABIC_STOPWORDS)}")
logger.info(f"   - Valid punctuation: {len(VALID_PUNCTUATION)}")

# %%
# ============================================================================
# SECTION 5: DATA LOADING UTILITIES
# ============================================================================

def iter_dataset_lines(dataset_dir: str, encoding: str = "utf-8") -> Generator[str, None, None]:
    """
    Iterate over all dataset files as a single line stream.
    
    This function implements lazy loading to handle large datasets
    that cannot fit in memory.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to directory containing .txt files
    encoding : str
        File encoding (default: utf-8)
        
    Yields:
    -------
    str
        One sentence/line at a time (stripped of newline)
    """
    # Get all text files sorted by name
    txt_files = sorted(Path(dataset_dir).glob("*.txt"))
    
    if not txt_files:
        logger.error(f"No .txt files found in {dataset_dir}")
        return
    
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line in f:
                    yield line.rstrip("\n")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue


def count_total_lines(dataset_dir: str) -> int:
    """
    Count total lines in dataset (for progress bar).
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
        
    Returns:
    --------
    int
        Total number of lines
    """
    total = 0
    for file_path in sorted(Path(dataset_dir).glob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def get_sample_lines(dataset_dir: str, n: int = 1000, seed: int = 42) -> List[str]:
    """
    Get random sample of lines for inspection.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
    n : int
        Number of samples
    seed : int
        Random seed
        
    Returns:
    --------
    List[str]
        Sample lines
    """
    random.seed(seed)
    
    # Collect lines from beginning for sampling
    lines = []
    for i, line in enumerate(iter_dataset_lines(dataset_dir)):
        if i >= n * 10:  # Get more than needed for random selection
            break
        if line.strip():
            lines.append(line)
    
    return random.sample(lines, min(n, len(lines)))


logger.info("✅ Data loading utilities ready!")

# %% [markdown]
# ---
# ## 2. Part 1: Problem Inspection (Before Preprocessing)
# 
# Before applying any preprocessing, we need to systematically identify and quantify
# all issues in the raw data. This helps us:
# 
# 1. Understand the scope of each problem
# 2. Prioritize preprocessing steps
# 3. Verify that preprocessing fixes the issues

# %% [markdown]
# ### 2.1 Character-Level Issues

# %%
# ============================================================================
# INSPECTION 2.1: CHARACTER-LEVEL ISSUES
# ============================================================================

def inspect_character_issues(dataset_dir: str, sample_size: int = 500000) -> Dict:
    """
    Inspect character-level issues in the dataset.
    
    Checks for:
    - Diacritics (tashkeel)
    - Out-of-vocabulary characters
    - Latin letters
    - Special characters
    - Alef variations
    - Tatweel (elongation)
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
    sample_size : int
        Number of lines to inspect
        
    Returns:
    --------
    Dict
        Dictionary containing issue statistics
    """
    logger.section("🔍 CHARACTER-LEVEL ISSUE INSPECTION")
    logger.info(f"Inspecting {sample_size:,} lines...")
    
    # Initialize counters
    stats = {
        'total_chars': 0,
        'total_lines': 0,
        'diacritics': Counter(),
        'latin_letters': Counter(),
        'oov_chars': Counter(),
        'alef_variations': Counter(),
        'tatweel_count': 0,
        'lines_with_diacritics': 0,
        'lines_with_latin': 0,
        'lines_with_oov': 0,
    }
    
    # Define valid character set
    valid_chars = set()
    valid_chars.update(ARABIC_LETTERS_EXTENDED)
    valid_chars.update(ARABIC_NUMERALS)
    valid_chars.update(WESTERN_NUMERALS)
    valid_chars.update(VALID_PUNCTUATION)
    valid_chars.update(LATIN_TO_ARABIC_PUNCT.keys())
    valid_chars.update(' \t\n')  # Whitespace
    valid_chars.update('()[]{}«»""\'-–—')  # Brackets and quotes
    valid_chars.update(ARABIC_DIACRITICS.keys())  # Diacritics (to count)
    
    # Latin letter pattern
    latin_pattern = re.compile(r'[A-Za-z]')
    
    # Create iterator
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="Inspecting characters")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        stats['total_chars'] += len(line)
        
        has_diacritics = False
        has_latin = False
        has_oov = False
        
        for char in line:
            # Check for diacritics
            if char in ARABIC_DIACRITICS:
                stats['diacritics'][char] += 1
                has_diacritics = True
            
            # Check for Latin letters
            if latin_pattern.match(char):
                stats['latin_letters'][char] += 1
                has_latin = True
            
            # Check for Alef variations
            if char in ALEF_VARIATIONS:
                stats['alef_variations'][char] += 1
            
            # Check for Tatweel
            if char == '\u0640':
                stats['tatweel_count'] += 1
            
            # Check for OOV characters
            if char not in valid_chars and not latin_pattern.match(char):
                stats['oov_chars'][char] += 1
                has_oov = True
        
        if has_diacritics:
            stats['lines_with_diacritics'] += 1
        if has_latin:
            stats['lines_with_latin'] += 1
        if has_oov:
            stats['lines_with_oov'] += 1
    
    # Display results
    logger.subsection("Diacritics (Tashkeel)")
    total_diacritics = sum(stats['diacritics'].values())
    logger.info(f"Total diacritics found: {total_diacritics:,}")
    logger.info(f"Lines with diacritics: {stats['lines_with_diacritics']:,} ({stats['lines_with_diacritics']/stats['total_lines']*100:.2f}%)")
    
    if stats['diacritics']:
        logger.info("Diacritic distribution:")
        for char, count in stats['diacritics'].most_common():
            name = ARABIC_DIACRITICS.get(char, 'Unknown')
            logger.info(f"   {repr(char)} ({name}): {count:,}")
    
    logger.subsection("Latin Letters")
    total_latin = sum(stats['latin_letters'].values())
    logger.info(f"Total Latin letters: {total_latin:,}")
    logger.info(f"Lines with Latin: {stats['lines_with_latin']:,} ({stats['lines_with_latin']/stats['total_lines']*100:.2f}%)")
    
    if stats['latin_letters']:
        logger.info("Top Latin letters:")
        for char, count in stats['latin_letters'].most_common(10):
            logger.info(f"   '{char}': {count:,}")
    
    logger.subsection("Alef Variations")
    total_alef_var = sum(stats['alef_variations'].values())
    logger.info(f"Total Alef variations: {total_alef_var:,}")
    
    if stats['alef_variations']:
        for char, count in stats['alef_variations'].most_common():
            logger.info(f"   '{char}': {count:,}")
    
    logger.subsection("Tatweel (Elongation)")
    logger.info(f"Tatweel count: {stats['tatweel_count']:,}")
    
    logger.subsection("Out-of-Vocabulary Characters")
    total_oov = sum(stats['oov_chars'].values())
    logger.info(f"Total OOV characters: {total_oov:,}")
    logger.info(f"Lines with OOV: {stats['lines_with_oov']:,} ({stats['lines_with_oov']/stats['total_lines']*100:.2f}%)")
    
    if stats['oov_chars']:
        logger.info("Top OOV characters:")
        for char, count in stats['oov_chars'].most_common(20):
            try:
                char_name = f"U+{ord(char):04X}"
            except:
                char_name = "Unknown"
            logger.info(f"   {repr(char)} ({char_name}): {count:,}")
    
    return stats


# Run character inspection
char_issues = inspect_character_issues(config.input_dir, sample_size=500000)

# %% [markdown]
# ### 2.2 Punctuation Issues

# %%
# ============================================================================
# INSPECTION 2.2: PUNCTUATION ISSUES
# ============================================================================

def inspect_punctuation_issues(dataset_dir: str, sample_size: int = 500000) -> Dict:
    """
    Inspect punctuation-related issues.
    
    Checks for:
    - Mixed Arabic/Latin punctuation
    - Consecutive punctuation
    - Missing spacing around punctuation
    - Invalid punctuation marks
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
    sample_size : int
        Number of lines to inspect
        
    Returns:
    --------
    Dict
        Dictionary containing issue statistics
    """
    logger.section("🔍 PUNCTUATION ISSUE INSPECTION")
    logger.info(f"Inspecting {sample_size:,} lines...")
    
    stats = {
        'total_lines': 0,
        'arabic_punct': Counter(),
        'latin_punct': Counter(),
        'other_punct': Counter(),
        'consecutive_punct': [],  # Store examples
        'consecutive_punct_count': 0,
        'attached_punct_count': 0,
        'lines_with_mixed_punct': 0,
    }
    
    # All punctuation for detection
    all_punct = set(ARABIC_PUNCTUATION.keys()) | set(LATIN_TO_ARABIC_PUNCT.keys())
    
    # Pattern for consecutive punctuation
    consecutive_pattern = re.compile(r'[،؛؟.,:;?!]{2,}')
    
    # Pattern for attached punctuation (no space before/after)
    # Arabic word followed immediately by punctuation with no space
    attached_pattern = re.compile(r'[\u0600-\u06FF][،؛؟.:!][\u0600-\u06FF]')
    
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="Inspecting punctuation")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        
        has_arabic_punct = False
        has_latin_punct = False
        
        # Count punctuation types
        for char in line:
            if char in ARABIC_PUNCTUATION:
                stats['arabic_punct'][char] += 1
                has_arabic_punct = True
            elif char in LATIN_TO_ARABIC_PUNCT:
                stats['latin_punct'][char] += 1
                has_latin_punct = True
            elif char in '()[]{}«»""\'':
                stats['other_punct'][char] += 1
        
        # Check for mixed punctuation
        if has_arabic_punct and has_latin_punct:
            stats['lines_with_mixed_punct'] += 1
        
        # Check for consecutive punctuation
        consecutive_matches = consecutive_pattern.findall(line)
        if consecutive_matches:
            stats['consecutive_punct_count'] += len(consecutive_matches)
            if len(stats['consecutive_punct']) < 20:  # Store examples
                for match in consecutive_matches:
                    stats['consecutive_punct'].append((match, line[:100]))
        
        # Check for attached punctuation
        attached_matches = attached_pattern.findall(line)
        if attached_matches:
            stats['attached_punct_count'] += len(attached_matches)
    
    # Display results
    logger.subsection("Arabic Punctuation")
    for char, count in stats['arabic_punct'].most_common():
        name = ARABIC_PUNCTUATION.get(char, 'Unknown')
        logger.info(f"   '{char}' ({name}): {count:,}")
    
    logger.subsection("Latin Punctuation (needs normalization)")
    for char, count in stats['latin_punct'].most_common():
        logger.info(f"   '{char}': {count:,} → should become '{LATIN_TO_ARABIC_PUNCT.get(char, char)}'")
    
    logger.subsection("Mixed Punctuation Lines")
    logger.info(f"Lines with mixed Arabic/Latin punctuation: {stats['lines_with_mixed_punct']:,}")
    logger.info(f"Percentage: {stats['lines_with_mixed_punct']/stats['total_lines']*100:.2f}%")
    
    logger.subsection("Consecutive Punctuation")
    logger.info(f"Total occurrences: {stats['consecutive_punct_count']:,}")
    if stats['consecutive_punct']:
        logger.info("Examples:")
        for match, context in stats['consecutive_punct'][:5]:
            logger.info(f"   '{match}' in: {context[:60]}...")
    
    logger.subsection("Attached Punctuation")
    logger.info(f"Cases where punctuation lacks proper spacing: {stats['attached_punct_count']:,}")
    
    return stats


# Run punctuation inspection
punct_issues = inspect_punctuation_issues(config.input_dir, sample_size=500000)

# %% [markdown]
# ### 2.3 Sentence-Level Issues

# %%
# ============================================================================
# INSPECTION 2.3: SENTENCE-LEVEL ISSUES
# ============================================================================

def inspect_sentence_issues(dataset_dir: str, sample_size: int = 1000000) -> Dict:
    """
    Inspect sentence-level issues.
    
    Checks for:
    - Empty lines
    - Very short sentences
    - Very long sentences
    - Missing terminal punctuation
    - Multiple sentences in one line
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
    sample_size : int
        Number of lines to inspect
        
    Returns:
    --------
    Dict
        Dictionary containing issue statistics
    """
    logger.section("🔍 SENTENCE-LEVEL ISSUE INSPECTION")
    logger.info(f"Inspecting {sample_size:,} lines...")
    
    stats = {
        'total_lines': 0,
        'empty_lines': 0,
        'very_short': [],  # < 3 words
        'very_long': [],   # > 100 words
        'word_counts': [],
        'missing_terminal': 0,
        'wrong_terminal': Counter(),
        'multiple_terminals': 0,
    }
    
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="Inspecting sentences")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        
        # Check empty
        stripped = line.strip()
        if not stripped:
            stats['empty_lines'] += 1
            continue
        
        # Count words
        words = stripped.split()
        word_count = len(words)
        stats['word_counts'].append(word_count)
        
        # Check very short
        if word_count < 3:
            if len(stats['very_short']) < 20:  # Store examples
                stats['very_short'].append(stripped)
        
        # Check very long
        if word_count > 100:
            if len(stats['very_long']) < 10:
                stats['very_long'].append((word_count, stripped[:100] + "..."))
        
        # Check terminal punctuation
        if stripped:
            last_char = stripped[-1]
            if last_char not in SENTENCE_TERMINALS:
                stats['missing_terminal'] += 1
                stats['wrong_terminal'][last_char] += 1
        
        # Check for multiple sentence terminals within line
        terminal_count = sum(1 for c in stripped[:-1] if c in SENTENCE_TERMINALS)
        if terminal_count > 0:
            stats['multiple_terminals'] += 1
    
    # Display results
    logger.subsection("Empty Lines")
    logger.info(f"Empty lines: {stats['empty_lines']:,} ({stats['empty_lines']/stats['total_lines']*100:.4f}%)")
    
    logger.subsection("Sentence Length Statistics")
    if stats['word_counts']:
        word_arr = np.array(stats['word_counts'])
        logger.info(f"Mean: {np.mean(word_arr):.2f} words")
        logger.info(f"Median: {np.median(word_arr):.2f} words")
        logger.info(f"Min: {np.min(word_arr)} words")
        logger.info(f"Max: {np.max(word_arr)} words")
        logger.info(f"Std: {np.std(word_arr):.2f} words")
    
    logger.subsection("Very Short Sentences (<3 words)")
    short_count = sum(1 for w in stats['word_counts'] if w < 3)
    logger.info(f"Count: {short_count:,} ({short_count/stats['total_lines']*100:.2f}%)")
    if stats['very_short']:
        logger.info("Examples:")
        for example in stats['very_short'][:5]:
            logger.info(f"   '{example}'")
    
    logger.subsection("Very Long Sentences (>100 words)")
    long_count = sum(1 for w in stats['word_counts'] if w > 100)
    logger.info(f"Count: {long_count:,} ({long_count/stats['total_lines']*100:.2f}%)")
    if stats['very_long']:
        logger.info("Examples:")
        for wc, example in stats['very_long'][:3]:
            logger.info(f"   [{wc} words] {example}")
    
    logger.subsection("Terminal Punctuation Issues")
    logger.info(f"Lines missing standard terminal: {stats['missing_terminal']:,}")
    if stats['wrong_terminal']:
        logger.info("Non-standard terminals found:")
        for char, count in stats['wrong_terminal'].most_common(10):
            logger.info(f"   '{char}': {count:,}")
    
    logger.subsection("Multiple Sentence Terminals")
    logger.info(f"Lines with terminal punctuation mid-sentence: {stats['multiple_terminals']:,}")
    
    return stats


# Run sentence inspection
sentence_issues = inspect_sentence_issues(config.input_dir, sample_size=1000000)

# %% [markdown]
# ### 2.4 Special Pattern Issues

# %%
# ============================================================================
# INSPECTION 2.4: SPECIAL PATTERN ISSUES
# ============================================================================

def inspect_special_patterns(dataset_dir: str, sample_size: int = 500000) -> Dict:
    """
    Inspect special patterns that might need handling.
    
    Checks for:
    - Waw conjunction attached to words
    - Numbers (Arabic and Western)
    - Document references (e.g., A/47/10)
    - URLs or email-like patterns
    - Repeated characters
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
    sample_size : int
        Number of lines to inspect
        
    Returns:
    --------
    Dict
        Dictionary containing pattern statistics
    """
    logger.section("🔍 SPECIAL PATTERN INSPECTION")
    logger.info(f"Inspecting {sample_size:,} lines...")
    
    stats = {
        'total_lines': 0,
        'waw_attached': Counter(),  # Words starting with و
        'arabic_numbers': 0,
        'western_numbers': 0,
        'mixed_number_lines': 0,
        'doc_references': [],
        'repeated_chars': [],
        'foreign_words': Counter(),
    }
    
    # Patterns
    waw_word_pattern = re.compile(r'\bو[\u0600-\u06FF]{2,}\b')
    arabic_num_pattern = re.compile(r'[٠-٩]+')
    western_num_pattern = re.compile(r'[0-9]+')
    doc_ref_pattern = re.compile(r'[A-Z]/[0-9]+(?:/[0-9A-Z]+)*')
    repeated_pattern = re.compile(r'(.)\1{3,}')  # Same char 4+ times
    foreign_word_pattern = re.compile(r'\b[A-Za-z]{3,}\b')  # 3+ Latin letters
    
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="Inspecting patterns")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        
        # Waw-attached words
        waw_matches = waw_word_pattern.findall(line)
        for match in waw_matches:
            stats['waw_attached'][match] += 1
        
        # Number systems
        has_arabic = bool(arabic_num_pattern.search(line))
        has_western = bool(western_num_pattern.search(line))
        
        if has_arabic:
            stats['arabic_numbers'] += len(arabic_num_pattern.findall(line))
        if has_western:
            stats['western_numbers'] += len(western_num_pattern.findall(line))
        if has_arabic and has_western:
            stats['mixed_number_lines'] += 1
        
        # Document references
        doc_refs = doc_ref_pattern.findall(line)
        if doc_refs and len(stats['doc_references']) < 20:
            stats['doc_references'].extend(doc_refs[:2])
        
        # Repeated characters
        repeated = repeated_pattern.findall(line)
        if repeated and len(stats['repeated_chars']) < 10:
            stats['repeated_chars'].append(line[:80])
        
        # Foreign words
        foreign = foreign_word_pattern.findall(line)
        for word in foreign:
            stats['foreign_words'][word] += 1
    
    # Display results
    logger.subsection("Waw Conjunction Patterns")
    logger.info(f"Total words starting with و: {sum(stats['waw_attached'].values()):,}")
    logger.info("Most common وَ-attached words:")
    for word, count in stats['waw_attached'].most_common(15):
        logger.info(f"   '{word}': {count:,}")
    
    logger.subsection("Number Systems")
    logger.info(f"Arabic numerals found: {stats['arabic_numbers']:,}")
    logger.info(f"Western numerals found: {stats['western_numbers']:,}")
    logger.info(f"Lines with mixed numerals: {stats['mixed_number_lines']:,}")
    
    logger.subsection("Document References")
    logger.info(f"Examples: {stats['doc_references'][:10]}")
    
    logger.subsection("Repeated Characters")
    if stats['repeated_chars']:
        logger.info("Lines with repeated characters:")
        for example in stats['repeated_chars'][:3]:
            logger.info(f"   {example}")
    else:
        logger.info("No significant repeated character patterns found")
    
    logger.subsection("Foreign Words")
    logger.info(f"Unique foreign words: {len(stats['foreign_words']):,}")
    logger.info("Most common foreign words:")
    for word, count in stats['foreign_words'].most_common(15):
        logger.info(f"   '{word}': {count:,}")
    
    return stats


# Run special pattern inspection
special_issues = inspect_special_patterns(config.input_dir, sample_size=500000)

# %% [markdown]
# ---
# ## 3. Part 2: Mandatory Preprocessing Steps
# 
# These preprocessing steps are **always applied** as they address fundamental
# data quality issues that would negatively impact model training.

# %% [markdown]
# ### 3.1 Remove Diacritics (Tashkeel)

# %%
# ============================================================================
# PREPROCESSING 3.1: REMOVE DIACRITICS
# ============================================================================

def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.
    
    Diacritics removed:
    - Fathatan (ً)
    - Dammatan (ٌ)
    - Kasratan (ٍ)
    - Fatha (َ)
    - Damma (ُ)
    - Kasra (ِ)
    - Shadda (ّ)
    - Sukun (ْ)
    
    Parameters:
    -----------
    text : str
        Input text with potential diacritics
        
    Returns:
    --------
    str
        Text with diacritics removed
        
    Example:
    --------
    >>> remove_diacritics("الْعَرَبِيَّة")
    'العربية'
    """
    return DIACRITICS_PATTERN.sub('', text)


# Test the function
logger.section("🔧 PREPROCESSING: Remove Diacritics")

test_cases = [
    "الْعَرَبِيَّة",
    "مُحَمَّد",
    "الأُمَمُ المُتَّحِدَة",
    "text without diacritics",
]

logger.info("Test cases:")
for test in test_cases:
    result = remove_diacritics(test)
    logger.info(f"   '{test}' → '{result}'")

# %% [markdown]
# ### 3.2 Normalize Alef Variations

# %%
# ============================================================================
# PREPROCESSING 3.2: NORMALIZE ALEF VARIATIONS
# ============================================================================

# Compile pattern for efficiency
ALEF_PATTERN = re.compile(r'[أإآٱ]')

def normalize_alef(text: str) -> str:
    """
    Normalize all Alef variations to bare Alef (ا).
    
    Normalizations:
    - أ (Alef with Hamza Above) → ا
    - إ (Alef with Hamza Below) → ا
    - آ (Alef with Madda) → ا
    - ٱ (Alef Wasla) → ا
    
    Parameters:
    -----------
    text : str
        Input text with potential Alef variations
        
    Returns:
    --------
    str
        Text with normalized Alef
        
    Example:
    --------
    >>> normalize_alef("أحمد إبراهيم آدم")
    'احمد ابراهيم ادم'
    """
    return ALEF_PATTERN.sub('ا', text)


# Test the function
logger.section("🔧 PREPROCESSING: Normalize Alef")

test_cases = [
    "أحمد",
    "إبراهيم",
    "آدم",
    "الأمم",
    "الإنسان",
]

logger.info("Test cases:")
for test in test_cases:
    result = normalize_alef(test)
    logger.info(f"   '{test}' → '{result}'")

# %% [markdown]
# ### 3.3 Normalize Teh Marbuta and Alef Maksura

# %%
# ============================================================================
# PREPROCESSING 3.3: NORMALIZE TEH MARBUTA AND ALEF MAKSURA
# ============================================================================

def normalize_teh_marbuta(text: str, to_heh: bool = False) -> str:
    """
    Handle Teh Marbuta (ة).
    
    Options:
    - Keep as is (default for this task)
    - Normalize to Heh (ه) - some NLP applications do this
    
    Parameters:
    -----------
    text : str
        Input text
    to_heh : bool
        If True, convert ة to ه
        
    Returns:
    --------
    str
        Processed text
    """
    if to_heh:
        return text.replace('ة', 'ه')
    return text


def normalize_alef_maksura(text: str) -> str:
    """
    Normalize Alef Maksura (ى) to Yeh (ي).
    
    Note: In some contexts, ى is kept distinct. For punctuation
    prediction, normalizing improves consistency.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with ى → ي
        
    Example:
    --------
    >>> normalize_alef_maksura("على")
    'علي'
    """
    return text.replace('ى', 'ي')


# Test the functions
logger.section("🔧 PREPROCESSING: Normalize Teh Marbuta & Alef Maksura")

test_cases = [
    ("مدرسة", "Teh Marbuta"),
    ("على", "Alef Maksura"),
    ("مستشفى", "Alef Maksura"),
    ("القاهرة", "Teh Marbuta"),
]

logger.info("Test cases:")
for test, note in test_cases:
    result_tm = normalize_teh_marbuta(test, to_heh=False)
    result_am = normalize_alef_maksura(test)
    logger.info(f"   '{test}' ({note})")
    logger.info(f"      Teh Marbuta (keep): '{result_tm}'")
    logger.info(f"      Alef Maksura → Yeh: '{result_am}'")

# %% [markdown]
# ### 3.4 Remove Out-of-Vocabulary Characters

# %%
# ============================================================================
# PREPROCESSING 3.4: REMOVE OUT-OF-VOCABULARY CHARACTERS
# ============================================================================

def build_valid_charset() -> set:
    """
    Build the set of valid characters for Arabic punctuation task.
    
    Valid characters include:
    - Arabic letters (including extended)
    - Arabic numerals
    - Valid punctuation marks
    - Whitespace
    
    Returns:
    --------
    set
        Set of valid characters
    """
    valid = set()
    
    # Arabic letters (basic + extended)
    valid.update('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوي')
    valid.update('ىپچژگڤ')  # Extended
    
    # Arabic numerals
    valid.update('٠١٢٣٤٥٦٧٨٩')
    
    # Valid punctuation (Arabic)
    valid.update('،؛؟.:!')
    
    # Basic structural characters
    valid.update(' ')  # Space
    
    # Keep some brackets for structure (will be handled later if needed)
    valid.update('()[]')
    
    return valid


VALID_CHARSET = build_valid_charset()


def remove_oov_characters(text: str, valid_chars: set = None, replacement: str = '') -> str:
    """
    Remove characters not in the valid character set.
    
    Parameters:
    -----------
    text : str
        Input text
    valid_chars : set
        Set of valid characters (uses VALID_CHARSET if None)
    replacement : str
        Character to replace OOV chars with (default: remove)
        
    Returns:
    --------
    str
        Text with OOV characters removed
    """
    if valid_chars is None:
        valid_chars = VALID_CHARSET
    
    result = []
    for char in text:
        if char in valid_chars:
            result.append(char)
        elif replacement:
            result.append(replacement)
    
    return ''.join(result)


# Test the function
logger.section("🔧 PREPROCESSING: Remove OOV Characters")

test_cases = [
    "النص العربي مع English text",
    "رقم: ١٢٣ و 456",
    "مع رموز خاصة: @#$%",
    "«نص بين أقواس»",
]

logger.info(f"Valid charset size: {len(VALID_CHARSET)}")
logger.info("Test cases:")
for test in test_cases:
    result = remove_oov_characters(test)
    logger.info(f"   '{test}'")
    logger.info(f"   → '{result}'")

# %% [markdown]
# ### 3.5 Remove Latin Letters

# %%
# ============================================================================
# PREPROCESSING 3.5: REMOVE LATIN LETTERS
# ============================================================================

LATIN_PATTERN = re.compile(r'[A-Za-z]+')

def remove_latin_letters(text: str, replacement: str = '') -> str:
    """
    Remove Latin letters from text.
    
    This handles:
    - Standalone English words
    - Document references (A/47/10 → /47/10)
    - Mixed text
    
    Parameters:
    -----------
    text : str
        Input text
    replacement : str
        Replacement for Latin sequences (default: remove)
        
    Returns:
    --------
    str
        Text without Latin letters
    """
    return LATIN_PATTERN.sub(replacement, text)


# Test the function
logger.section("🔧 PREPROCESSING: Remove Latin Letters")

test_cases = [
    "الأمم المتحدة United Nations",
    "الوثيقة A/47/10",
    "برنامج UNDP للتنمية",
    "add. 1",
]

logger.info("Test cases:")
for test in test_cases:
    result = remove_latin_letters(test)
    logger.info(f"   '{test}' → '{result}'")

# %% [markdown]
# ### 3.6 Unify Numbers (Arabic Numerals)

# %%
# ============================================================================
# PREPROCESSING 3.6: UNIFY NUMBERS TO ARABIC
# ============================================================================

def unify_numbers_to_arabic(text: str) -> str:
    """
    Convert all Western numerals (0-9) to Arabic numerals (٠-٩).
    
    Parameters:
    -----------
    text : str
        Input text with potential Western numerals
        
    Returns:
    --------
    str
        Text with Arabic numerals only
        
    Example:
    --------
    >>> unify_numbers_to_arabic("عام 2024")
    'عام ٢٠٢٤'
    """
    return text.translate(WESTERN_TO_ARABIC_NUMS)


def unify_numbers_to_western(text: str) -> str:
    """
    Convert all Arabic numerals (٠-٩) to Western numerals (0-9).
    
    Alternative approach - some models prefer Western numerals.
    
    Parameters:
    -----------
    text : str
        Input text with potential Arabic numerals
        
    Returns:
    --------
    str
        Text with Western numerals only
    """
    return text.translate(ARABIC_TO_WESTERN_NUMS)


# Test the function
logger.section("🔧 PREPROCESSING: Unify Numbers")

test_cases = [
    "عام 2024",
    "رقم ١٢٣",
    "مبلغ 100 دولار",
    "٥٠٠ + 500 = ١٠٠٠",
]

logger.info("Test cases (→ Arabic numerals):")
for test in test_cases:
    result = unify_numbers_to_arabic(test)
    logger.info(f"   '{test}' → '{result}'")

# %% [markdown]
# ### 3.7 Unify Punctuation (Arabic Punctuation)

# %%
# ============================================================================
# PREPROCESSING 3.7: UNIFY PUNCTUATION TO ARABIC
# ============================================================================

def unify_punctuation_to_arabic(text: str) -> str:
    """
    Convert Latin punctuation to Arabic equivalents.
    
    Conversions:
    - , → ،  (comma)
    - ; → ؛  (semicolon)
    - ? → ؟  (question mark)
    
    Note: Period (.), colon (:), and exclamation (!) remain unchanged
    as they are used in both systems.
    
    Parameters:
    -----------
    text : str
        Input text with potential Latin punctuation
        
    Returns:
    --------
    str
        Text with Arabic punctuation
    """
    for latin, arabic in LATIN_TO_ARABIC_PUNCT.items():
        text = text.replace(latin, arabic)
    return text


# Test the function
logger.section("🔧 PREPROCESSING: Unify Punctuation")

test_cases = [
    "أولاً, ثانياً, ثالثاً",
    "هل هذا صحيح?",
    "ملاحظة; هذا مهم",
    "النص يحتوي على , و ; و ?",
]

logger.info("Test cases:")
for test in test_cases:
    result = unify_punctuation_to_arabic(test)
    logger.info(f"   '{test}'")
    logger.info(f"   → '{result}'")

# %% [markdown]
# ### 3.8 Handle Consecutive Punctuation

# %%
# ============================================================================
# PREPROCESSING 3.8: HANDLE CONSECUTIVE PUNCTUATION
# ============================================================================

def handle_consecutive_punctuation(text: str, keep_first: bool = True) -> str:
    """
    Handle consecutive punctuation marks.
    
    Strategies:
    - keep_first: Keep the first punctuation, remove rest
    - keep_last: Keep the last punctuation, remove rest
    - keep_strongest: Keep the "strongest" (. > ? > ! > ; > , > :)
    
    Parameters:
    -----------
    text : str
        Input text
    keep_first : bool
        If True, keep first punctuation in sequence
        
    Returns:
    --------
    str
        Text with consecutive punctuation handled
    """
    # Pattern matches 2+ punctuation marks in sequence
    punct_chars = r'،؛؟.,:;?!'
    pattern = re.compile(f'([{punct_chars}])([{punct_chars}]+)')
    
    if keep_first:
        # Keep first, remove subsequent
        return pattern.sub(r'\1', text)
    else:
        # Keep last
        def keep_last_match(m):
            return m.group(0)[-1]
        return pattern.sub(keep_last_match, text)


def handle_consecutive_punctuation_smart(text: str) -> str:
    """
    Smart handling of consecutive punctuation.
    
    Uses priority: . > ؟ > ! > ؛ > ، > :
    Keeps the highest priority punctuation.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with consecutive punctuation reduced
    """
    # Priority order (highest first)
    priority = {'.': 6, '؟': 5, '?': 5, '!': 4, '؛': 3, ';': 3, '،': 2, ',': 2, ':': 1}
    
    punct_chars = r'،؛؟.,:;?!'
    pattern = re.compile(f'[{punct_chars}]{{2,}}')
    
    def replace_func(match):
        sequence = match.group(0)
        # Find highest priority punctuation
        best_char = sequence[0]
        best_priority = priority.get(best_char, 0)
        
        for char in sequence:
            char_priority = priority.get(char, 0)
            if char_priority > best_priority:
                best_char = char
                best_priority = char_priority
        
        return best_char
    
    return pattern.sub(replace_func, text)


# Test the function
logger.section("🔧 PREPROCESSING: Handle Consecutive Punctuation")

test_cases = [
    "ماذا؟؟؟",
    "هذا صحيح..",
    "أولاً،،",
    "انتهى.؟",
    "نهاية النص.,",
]

logger.info("Test cases:")
for test in test_cases:
    result_first = handle_consecutive_punctuation(test, keep_first=True)
    result_smart = handle_consecutive_punctuation_smart(test)
    logger.info(f"   '{test}'")
    logger.info(f"      Keep first: '{result_first}'")
    logger.info(f"      Smart:      '{result_smart}'")

# %% [markdown]
# ### 3.9 Normalize Whitespace and Punctuation Spacing

# %%
# ============================================================================
# PREPROCESSING 3.9: NORMALIZE WHITESPACE AND PUNCTUATION SPACING
# ============================================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    - Replace multiple spaces with single space
    - Remove leading/trailing whitespace
    - Replace tabs with spaces
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with normalized whitespace
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def add_punctuation_spacing(text: str) -> str:
    """
    Ensure proper spacing around punctuation marks.
    
    Rules:
    - Space after punctuation (if followed by letter/number)
    - No space before punctuation
    - Handle Arabic RTL properly
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with proper punctuation spacing
    """
    punct_marks = '،؛؟.:!'
    
    # Remove space before punctuation
    for p in punct_marks:
        text = re.sub(rf'\s+{re.escape(p)}', p, text)
    
    # Add space after punctuation if followed by Arabic letter or number
    for p in punct_marks:
        # After punct, if followed by Arabic char without space, add space
        text = re.sub(
            rf'{re.escape(p)}([\u0600-\u06FF٠-٩])',
            rf'{p} \1',
            text
        )
    
    # Clean up multiple spaces that might have been created
    text = re.sub(r' +', ' ', text)
    
    return text


# Test the functions
logger.section("🔧 PREPROCESSING: Normalize Whitespace & Punctuation Spacing")

test_cases = [
    "النص   مع   مسافات    كثيرة",
    "كلمة،كلمة",
    "نص .مع مسافة قبل النقطة",
    "سؤال؟جواب",
    "أولاً ، ثانياً ، ثالثاً",
]

logger.info("Test cases:")
for test in test_cases:
    result_ws = normalize_whitespace(test)
    result_spacing = add_punctuation_spacing(result_ws)
    logger.info(f"   '{test}'")
    logger.info(f"      After whitespace norm: '{result_ws}'")
    logger.info(f"      After punct spacing:   '{result_spacing}'")

# %% [markdown]
# ### 3.10 Remove Empty and Very Short Lines

# %%
# ============================================================================
# PREPROCESSING 3.10: REMOVE EMPTY AND VERY SHORT LINES
# ============================================================================

def is_valid_sentence(text: str, min_words: int = 3) -> bool:
    """
    Check if a sentence meets minimum requirements.
    
    Parameters:
    -----------
    text : str
        Input text (should be preprocessed)
    min_words : int
        Minimum number of words required
        
    Returns:
    --------
    bool
        True if sentence is valid
    """
    # Strip whitespace
    text = text.strip()
    
    # Check for empty
    if not text:
        return False
    
    # Count Arabic words only
    arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
    
    return len(arabic_words) >= min_words


def filter_sentence(text: str, min_words: int = 3) -> Optional[str]:
    """
    Filter and return sentence if valid, None otherwise.
    
    Parameters:
    -----------
    text : str
        Input text
    min_words : int
        Minimum number of words
        
    Returns:
    --------
    Optional[str]
        Sentence if valid, None otherwise
    """
    if is_valid_sentence(text, min_words):
        return text
    return None


# Test the function
logger.section("🔧 PREPROCESSING: Filter Short/Empty Sentences")

test_cases = [
    "",
    "نعم",
    "أ",
    "نعم لا",
    "هذا نص صحيح ومقبول",
    "   ",
    "1.",
    "نص قصير جدا",
]

logger.info(f"Minimum words required: {config.min_words}")
logger.info("Test cases:")
for test in test_cases:
    is_valid = is_valid_sentence(test, min_words=config.min_words)
    status = "✅ Valid" if is_valid else "❌ Invalid"
    logger.info(f"   '{test}' → {status}")

# %% [markdown]
# ### 3.11 Process Long Sentences

# %%
# ============================================================================
# PREPROCESSING 3.11: PROCESS LONG SENTENCES
# ============================================================================

def truncate_sentence(text: str, max_words: int = 100) -> str:
    """
    Truncate sentence to maximum word count.
    
    Strategy: Truncate at word boundary, try to end at punctuation.
    
    Parameters:
    -----------
    text : str
        Input text
    max_words : int
        Maximum number of words
        
    Returns:
    --------
    str
        Truncated text
    """
    words = text.split()
    
    if len(words) <= max_words:
        return text
    
    # Truncate to max_words
    truncated_words = words[:max_words]
    truncated = ' '.join(truncated_words)
    
    # Ensure ends with punctuation
    if truncated and truncated[-1] not in SENTENCE_TERMINALS:
        truncated += '.'
    
    return truncated


def split_long_sentence(text: str, max_words: int = 100) -> List[str]:
    """
    Split long sentence into multiple sentences at natural boundaries.
    
    Strategy: Split at punctuation marks (،؛) that create natural breaks.
    
    Parameters:
    -----------
    text : str
        Input text
    max_words : int
        Maximum words per segment
        
    Returns:
    --------
    List[str]
        List of sentence segments
    """
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    
    # Find potential split points (after ، or ؛)
    segments = []
    current_segment = []
    word_count = 0
    
    for word in words:
        current_segment.append(word)
        word_count += 1
        
        # Check if this word ends with comma/semicolon and we're past halfway
        if word_count >= max_words // 2:
            if word.endswith('،') or word.endswith('؛'):
                # Create segment
                segment_text = ' '.join(current_segment)
                segments.append(segment_text)
                current_segment = []
                word_count = 0
        
        # Force split if we hit max
        if word_count >= max_words:
            segment_text = ' '.join(current_segment)
            if not segment_text.endswith(('.', '؟', '!')):
                segment_text += '.'
            segments.append(segment_text)
            current_segment = []
            word_count = 0
    
    # Add remaining
    if current_segment:
        segment_text = ' '.join(current_segment)
        segments.append(segment_text)
    
    return segments


# Test the functions
logger.section("🔧 PREPROCESSING: Handle Long Sentences")

# Create a long test sentence
long_sentence = " ".join(["كلمة"] * 150)
logger.info(f"Original length: {len(long_sentence.split())} words")

truncated = truncate_sentence(long_sentence, max_words=100)
logger.info(f"After truncation: {len(truncated.split())} words")
logger.info(f"Ends with: '{truncated[-10:]}'")

# Test with natural breaks
long_with_punct = "هذا نص طويل يحتوي على فقرات متعددة، وكل فقرة تحتوي على معلومات مهمة، " * 20
segments = split_long_sentence(long_with_punct, max_words=50)
logger.info(f"\nSplit into {len(segments)} segments:")
for i, seg in enumerate(segments[:3]):
    logger.info(f"   Segment {i+1}: {len(seg.split())} words")

# %% [markdown]
# ---
# ## 4. Part 3: Optional Preprocessing Steps
# 
# These preprocessing steps are **optional** and can be toggled for experimentation.
# They may improve or degrade model performance depending on the task.

# %% [markdown]
# ### 4.1 Separate Waw Conjunction from Words

# %%
# ============================================================================
# OPTIONAL 4.1: SEPARATE WAW CONJUNCTION
# ============================================================================

def separate_waw_conjunction(text: str, min_remaining_length: int = 3) -> str:
    """
    Separate the conjunction و (waw) from the beginning of words.
    
    In Arabic, waw is often attached to the following word as a prefix
    meaning "and". Separating it can help with:
    - Better tokenization
    - Consistent word boundaries
    - Improved punctuation prediction before conjunctions
    
    Parameters:
    -----------
    text : str
        Input text
    min_remaining_length : int
        Only separate if remaining word has this many chars
        
    Returns:
    --------
    str
        Text with separated waw conjunctions
        
    Example:
    --------
    >>> separate_waw_conjunction("وقال الرجل وذهب")
    'و قال الرجل و ذهب'
    """
    # Words where waw is part of the root (should NOT be separated)
    waw_root_words = {
        'وقت', 'وجه', 'وضع', 'وصل', 'وقع', 'وزن', 'وفد', 'ورق', 'وطن',
        'وسط', 'وحدة', 'وزير', 'وزارة', 'ولاية', 'ولد', 'والد', 'والدة',
        'وثيقة', 'وثائق', 'واقع', 'واجب', 'وفاة', 'وكالة', 'وكيل',
        'واحد', 'واحدة', 'وسيلة', 'وسائل', 'ورشة', 'وظيفة', 'وظائف',
    }
    
    def should_separate(word: str) -> bool:
        """Check if waw should be separated from this word."""
        if not word.startswith('و'):
            return False
        
        if len(word) < min_remaining_length + 1:  # +1 for the waw
            return False
        
        # Check if word (with waw) is a root word
        if word in waw_root_words:
            return False
        
        # Check if word without waw exists as valid word
        # (This is a heuristic - ideally use a dictionary)
        remaining = word[1:]
        
        # Common prefixes that indicate waw is conjunction
        conjunction_indicators = [
            'ال',   # و + ال (definite article)
            'هو', 'هي', 'هم',  # pronouns
            'قد', 'لم', 'لن',  # particles
            'كان', 'يكون',     # verbs
            'أن', 'إن',        # particles
        ]
        
        for indicator in conjunction_indicators:
            if remaining.startswith(indicator):
                return True
        
        # If remaining word starts with definite article, likely conjunction
        if remaining.startswith('ال'):
            return True
        
        return len(remaining) >= min_remaining_length
    
    words = text.split()
    result = []
    
    for word in words:
        if should_separate(word):
            result.append('و')
            result.append(word[1:])
        else:
            result.append(word)
    
    return ' '.join(result)


# Test the function
logger.section("🔧 OPTIONAL: Separate Waw Conjunction")

test_cases = [
    "وقال الرجل",
    "والأمم المتحدة",
    "وفي هذا الصدد",
    "وقت الاجتماع",  # Should NOT separate (وقت is a root word)
    "وثيقة مهمة",   # Should NOT separate
    "والتنمية المستدامة",
    "وهو يعمل",
]

logger.info("Test cases:")
for test in test_cases:
    result = separate_waw_conjunction(test)
    logger.info(f"   '{test}'")
    logger.info(f"   → '{result}'")

# %% [markdown]
# ### 4.2 Stopword Handling Strategies

# %%
# ============================================================================
# OPTIONAL 4.2: STOPWORD HANDLING
# ============================================================================

def mark_stopwords(text: str, marker: str = '<SW>') -> str:
    """
    Mark stopwords with a special token (for analysis/experimentation).
    
    Parameters:
    -----------
    text : str
        Input text
    marker : str
        Marker to add after stopwords
        
    Returns:
    --------
    str
        Text with marked stopwords
    """
    words = text.split()
    result = []
    
    for word in words:
        # Remove punctuation for checking
        clean_word = re.sub(r'[،؛؟.:!]', '', word)
        
        if clean_word in ARABIC_STOPWORDS:
            result.append(word + marker)
        else:
            result.append(word)
    
    return ' '.join(result)


def remove_stopwords(text: str, keep_structure: bool = True) -> str:
    """
    Remove stopwords from text.
    
    WARNING: This may hurt punctuation prediction as stopwords
    provide important structural context.
    
    Parameters:
    -----------
    text : str
        Input text
    keep_structure : bool
        If True, keep punctuation even if attached to stopwords
        
    Returns:
    --------
    str
        Text without stopwords
    """
    words = text.split()
    result = []
    
    for word in words:
        # Separate word from trailing punctuation
        punct = ''
        clean_word = word
        
        if word and word[-1] in '،؛؟.:!':
            punct = word[-1]
            clean_word = word[:-1]
        
        if clean_word not in ARABIC_STOPWORDS:
            result.append(word)
        elif keep_structure and punct:
            # Keep punctuation even if word is stopword
            if result:
                result[-1] += punct
    
    return ' '.join(result)


# Test the function
logger.section("🔧 OPTIONAL: Stopword Handling")

test_cases = [
    "وفي هذا الصدد",
    "من أجل التنمية",
    "الأمم المتحدة هي منظمة دولية",
]

logger.info("Test cases:")
for test in test_cases:
    marked = mark_stopwords(test)
    removed = remove_stopwords(test)
    logger.info(f"   Original: '{test}'")
    logger.info(f"   Marked:   '{marked}'")
    logger.info(f"   Removed:  '{removed}'")

# %% [markdown]
# ### 4.3 Number Token Replacement

# %%
# ============================================================================
# OPTIONAL 4.3: NUMBER TOKEN REPLACEMENT
# ============================================================================

def replace_numbers_with_token(text: str, token: str = '<NUM>') -> str:
    """
    Replace all numbers with a special token.
    
    This can help:
    - Reduce vocabulary size
    - Focus model on structure rather than specific numbers
    
    Parameters:
    -----------
    text : str
        Input text
    token : str
        Token to replace numbers with
        
    Returns:
    --------
    str
        Text with numbers replaced
    """
    # Pattern for Arabic or Western numerals
    number_pattern = re.compile(r'[٠-٩0-9]+')
    return number_pattern.sub(token, text)


def normalize_number_format(text: str) -> str:
    """
    Normalize number formatting (e.g., thousands separators).
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with normalized number formats
    """
    # Remove thousands separators (both , and ٬)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'([\u0660-\u0669])٬([\u0660-\u0669])', r'\1\2', text)
    
    return text


# Test the function
logger.section("🔧 OPTIONAL: Number Token Replacement")

test_cases = [
    "في عام ٢٠٢٤",
    "مبلغ ١٠٠٠٠٠ دولار",
    "من ٥ إلى ١٠ سنوات",
    "القرار رقم ٤٧/١٢٣",
]

logger.info("Test cases:")
for test in test_cases:
    result = replace_numbers_with_token(test)
    logger.info(f"   '{test}'")
    logger.info(f"   → '{result}'")

# %% [markdown]
# ### 4.4 Rare Word Handling

# %%
# ============================================================================
# OPTIONAL 4.4: RARE WORD HANDLING
# ============================================================================

def build_vocabulary(dataset_dir: str, sample_size: int = 1000000) -> Counter:
    """
    Build vocabulary with word frequencies.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset
    sample_size : int
        Number of lines to process
        
    Returns:
    --------
    Counter
        Word frequency counter
    """
    vocab = Counter()
    arabic_word_pattern = re.compile(r'[\u0600-\u06FF]+')
    
    for i, line in enumerate(iter_dataset_lines(dataset_dir)):
        if i >= sample_size:
            break
        
        words = arabic_word_pattern.findall(line)
        vocab.update(words)
    
    return vocab


def replace_rare_words(text: str, vocab: Counter, threshold: int = 5, 
                       token: str = '<UNK>') -> str:
    """
    Replace rare words (below frequency threshold) with a special token.
    
    Parameters:
    -----------
    text : str
        Input text
    vocab : Counter
        Vocabulary with frequencies
    threshold : int
        Minimum frequency to keep word
    token : str
        Replacement token
        
    Returns:
    --------
    str
        Text with rare words replaced
    """
    arabic_word_pattern = re.compile(r'[\u0600-\u06FF]+')
    
    def replace_if_rare(match):
        word = match.group(0)
        if vocab.get(word, 0) < threshold:
            return token
        return word
    
    return arabic_word_pattern.sub(replace_if_rare, text)


# Note: Building vocabulary is expensive, so we'll demonstrate with a mock
logger.section("🔧 OPTIONAL: Rare Word Handling")

logger.info("Building vocabulary is computationally expensive.")
logger.info("In production, vocabulary should be built once and saved.")
logger.info("\nExample usage:")
logger.info("   vocab = build_vocabulary(dataset_dir, sample_size=1000000)")
logger.info("   text = replace_rare_words(text, vocab, threshold=5)")

# %% [markdown]
# ### 4.5 Sentence Length Normalization

# %%
# ============================================================================
# OPTIONAL 4.5: SENTENCE LENGTH NORMALIZATION
# ============================================================================

def pad_sentence(text: str, target_length: int, pad_token: str = '<PAD>') -> str:
    """
    Pad sentence to target length.
    
    Note: This is typically done during batching, not preprocessing.
    Including here for completeness.
    
    Parameters:
    -----------
    text : str
        Input text
    target_length : int
        Target number of words
    pad_token : str
        Token to use for padding
        
    Returns:
    --------
    str
        Padded text
    """
    words = text.split()
    
    if len(words) >= target_length:
        return text
    
    padding = [pad_token] * (target_length - len(words))
    return text + ' ' + ' '.join(padding)


def split_into_chunks(text: str, chunk_size: int = 50, overlap: int = 10) -> List[str]:
    """
    Split long text into overlapping chunks.
    
    Useful for very long documents where context needs to be preserved.
    
    Parameters:
    -----------
    text : str
        Input text
    chunk_size : int
        Target words per chunk
    overlap : int
        Words to overlap between chunks
        
    Returns:
    --------
    List[str]
        List of text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        # Move start with overlap
        start = end - overlap
        if start >= len(words) - overlap:
            break
    
    return chunks


# Test the functions
logger.section("🔧 OPTIONAL: Sentence Length Normalization")

test_text = "هذا نص طويل يحتوي على العديد من الكلمات التي تحتاج إلى معالجة"
logger.info(f"Original: {len(test_text.split())} words")

chunks = split_into_chunks(test_text, chunk_size=5, overlap=2)
logger.info(f"Split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    logger.info(f"   Chunk {i+1}: '{chunk}'")

# %% [markdown]
# ### 4.6 Remove/Replace Foreign Terms

# %%
# ============================================================================
# OPTIONAL 4.6: HANDLE FOREIGN TERMS
# ============================================================================

def remove_document_references(text: str) -> str:
    """
    Remove UN-style document references (e.g., A/47/10, S/RES/1234).
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text without document references
    """
    # Pattern for document references
    doc_pattern = re.compile(r'[A-Z]/[A-Z0-9]+(?:/[A-Z0-9]+)*')
    return doc_pattern.sub('', text)


def replace_foreign_with_token(text: str, token: str = '<FOREIGN>') -> str:
    """
    Replace foreign (non-Arabic) terms with a special token.
    
    Parameters:
    -----------
    text : str
        Input text
    token : str
        Replacement token
        
    Returns:
    --------
    str
        Text with foreign terms replaced
    """
    # Pattern for Latin words (3+ letters)
    foreign_pattern = re.compile(r'\b[A-Za-z]{3,}\b')
    return foreign_pattern.sub(token, text)


# Test the functions
logger.section("🔧 OPTIONAL: Handle Foreign Terms")

test_cases = [
    "الوثيقة A/47/10 المؤرخة",
    "برنامج UNDP للتنمية",
    "قرار مجلس الأمن S/RES/1234",
]

logger.info("Test cases:")
for test in test_cases:
    no_refs = remove_document_references(test)
    replaced = replace_foreign_with_token(test)
    logger.info(f"   Original: '{test}'")
    logger.info(f"   Remove refs: '{no_refs}'")
    logger.info(f"   Replace foreign: '{replaced}'")

# %% [markdown]
# ---
# ## 5. Part 4: Complete Preprocessing Pipeline

# %%
# ============================================================================
# SECTION 5: COMPLETE PREPROCESSING PIPELINE
# ============================================================================

@dataclass
class PreprocessingStats:
    """Statistics collected during preprocessing."""
    total_input_lines: int = 0
    total_output_lines: int = 0
    empty_lines_removed: int = 0
    short_lines_removed: int = 0
    long_lines_truncated: int = 0
    diacritics_removed: int = 0
    alef_normalized: int = 0
    punct_normalized: int = 0
    numbers_normalized: int = 0
    latin_removed: int = 0
    oov_removed: int = 0
    consecutive_punct_fixed: int = 0
    whitespace_fixed: int = 0


class ArabicTextPreprocessor:
    """
    Complete preprocessing pipeline for Arabic punctuation dataset.
    
    This class provides a configurable preprocessing pipeline that can be
    used with both mandatory and optional preprocessing steps.
    
    Attributes:
    -----------
    config : PreprocessingConfig
        Configuration object containing all settings
    stats : PreprocessingStats
        Statistics collected during preprocessing
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the preprocessor with configuration.
        
        Parameters:
        -----------
        config : PreprocessingConfig
            Configuration object
        """
        self.config = config
        self.stats = PreprocessingStats()
        self.vocab = None  # For rare word handling
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652]')
        self.alef_pattern = re.compile(r'[أإآٱ]')
        self.latin_pattern = re.compile(r'[A-Za-z]+')
        self.number_western = re.compile(r'[0-9]')
        self.consecutive_punct = re.compile(r'[،؛؟.,:;?!]{2,}')
        self.multi_space = re.compile(r' +')
        self.arabic_word = re.compile(r'[\u0600-\u06FF]+')
    
    def preprocess_line(self, text: str, apply_optional: bool = False) -> Optional[str]:
        """
        Apply full preprocessing pipeline to a single line.
        
        Parameters:
        -----------
        text : str
            Input text line
        apply_optional : bool
            Whether to apply optional preprocessing steps
            
        Returns:
        --------
        Optional[str]
            Preprocessed text, or None if line should be filtered
        """
        original = text
        
        # Track changes for statistics
        had_diacritics = bool(self.diacritics_pattern.search(text))
        had_alef_var = bool(self.alef_pattern.search(text))
        had_latin = bool(self.latin_pattern.search(text))
        had_consec_punct = bool(self.consecutive_punct.search(text))
        
        # ============================================
        # MANDATORY PREPROCESSING
        # ============================================
        
        # 1. Remove diacritics
        if self.config.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)
            if had_diacritics:
                self.stats.diacritics_removed += 1
        
        # 2. Normalize Alef
        if self.config.normalize_alef:
            text = self.alef_pattern.sub('ا', text)
            if had_alef_var:
                self.stats.alef_normalized += 1
        
        # 3. Normalize Alef Maksura (ى → ي)
        if self.config.normalize_alef_maksura:
            text = text.replace('ى', 'ي')
        
        # 4. Remove Tatweel
        if self.config.remove_tatweel:
            text = text.replace('\u0640', '')
        
        # 5. Unify punctuation to Arabic
        if self.config.unify_punctuation_to_arabic:
            for latin, arabic in LATIN_TO_ARABIC_PUNCT.items():
                if latin in text:
                    text = text.replace(latin, arabic)
                    self.stats.punct_normalized += 1
        
        # 6. Unify numbers to Arabic
        if self.config.unify_numbers_to_arabic:
            if self.number_western.search(text):
                text = text.translate(WESTERN_TO_ARABIC_NUMS)
                self.stats.numbers_normalized += 1
        
        # 7. Remove Latin letters
        if self.config.remove_latin_letters:
            if had_latin:
                text = self.latin_pattern.sub('', text)
                self.stats.latin_removed += 1
        
        # 8. Remove OOV characters
        if self.config.remove_oov_chars:
            original_len = len(text)
            text = remove_oov_characters(text)
            if len(text) < original_len:
                self.stats.oov_removed += 1
        
        # 9. Handle consecutive punctuation
        if self.config.handle_consecutive_punct:
            if had_consec_punct:
                text = handle_consecutive_punctuation_smart(text)
                self.stats.consecutive_punct_fixed += 1
        
        # 10. Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.multi_space.sub(' ', text)
            text = text.strip()
            self.stats.whitespace_fixed += 1
        
        # 11. Add punctuation spacing
        if self.config.add_punct_spacing:
            text = add_punctuation_spacing(text)
        
        # ============================================
        # OPTIONAL PREPROCESSING
        # ============================================
        
        if apply_optional:
            # Separate waw conjunction
            if self.config.separate_waw_conjunction:
                text = separate_waw_conjunction(text)
            
            # Replace numbers with token
            if self.config.replace_numbers_with_token:
                text = replace_numbers_with_token(text, '<NUM>')
            
            # Remove foreign terms
            if self.config.remove_foreign_terms:
                text = remove_document_references(text)
        
        # ============================================
        # FILTERING
        # ============================================
        
        # Final whitespace cleanup
        text = self.multi_space.sub(' ', text).strip()
        
        # Check for empty
        if self.config.remove_empty_lines and not text:
            self.stats.empty_lines_removed += 1
            return None
        
        # Check word count
        word_count = len(self.arabic_word.findall(text))
        
        # Filter short sentences
        if word_count < self.config.min_words:
            self.stats.short_lines_removed += 1
            return None
        
        # Handle long sentences
        if word_count > self.config.max_words:
            text = truncate_sentence(text, self.config.max_words)
            self.stats.long_lines_truncated += 1
        
        return text
    
    def process_dataset(self, input_dir: str, output_file: str, 
                        apply_optional: bool = False,
                        sample_size: Optional[int] = None) -> PreprocessingStats:
        """
        Process entire dataset and save to output file.
        
        Parameters:
        -----------
        input_dir : str
            Path to input dataset directory
        output_file : str
            Path to output file
        apply_optional : bool
            Whether to apply optional preprocessing
        sample_size : Optional[int]
            Limit processing to this many lines (None = all)
            
        Returns:
        --------
        PreprocessingStats
            Statistics from preprocessing
        """
        logger.section("🚀 PROCESSING DATASET")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Apply optional steps: {apply_optional}")
        
        # Reset statistics
        self.stats = PreprocessingStats()
        
        # Count total lines for progress bar
        if sample_size is None:
            total_lines = count_total_lines(input_dir)
            logger.info(f"Total lines to process: {total_lines:,}")
        else:
            total_lines = sample_size
            logger.info(f"Processing sample of {total_lines:,} lines")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process and write
        iterator = iter_dataset_lines(input_dir)
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=total_lines, desc="Preprocessing")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, line in enumerate(iterator):
                if sample_size and i >= sample_size:
                    break
                
                self.stats.total_input_lines += 1
                
                # Preprocess line
                processed = self.preprocess_line(line, apply_optional)
                
                # Write if not filtered
                if processed:
                    f.write(processed + '\n')
                    self.stats.total_output_lines += 1
        
        # Log statistics
        self._log_statistics()
        
        return self.stats
    
    def _log_statistics(self):
        """Log preprocessing statistics."""
        logger.section("📊 PREPROCESSING STATISTICS")
        
        logger.info(f"Input lines:  {self.stats.total_input_lines:,}")
        logger.info(f"Output lines: {self.stats.total_output_lines:,}")
        
        kept_pct = self.stats.total_output_lines / max(self.stats.total_input_lines, 1) * 100
        logger.info(f"Lines kept:   {kept_pct:.2f}%")
        
        logger.subsection("Filtering Statistics")
        logger.info(f"Empty lines removed:     {self.stats.empty_lines_removed:,}")
        logger.info(f"Short lines removed:     {self.stats.short_lines_removed:,}")
        logger.info(f"Long lines truncated:    {self.stats.long_lines_truncated:,}")
        
        logger.subsection("Normalization Statistics")
        logger.info(f"Diacritics removed:      {self.stats.diacritics_removed:,}")
        logger.info(f"Alef normalized:         {self.stats.alef_normalized:,}")
        logger.info(f"Punctuation normalized:  {self.stats.punct_normalized:,}")
        logger.info(f"Numbers normalized:      {self.stats.numbers_normalized:,}")
        logger.info(f"Latin removed:           {self.stats.latin_removed:,}")
        logger.info(f"OOV chars removed:       {self.stats.oov_removed:,}")
        logger.info(f"Consec. punct fixed:     {self.stats.consecutive_punct_fixed:,}")


# Instantiate preprocessor
preprocessor = ArabicTextPreprocessor(config)
logger.success("Preprocessor initialized!")

# %% [markdown]
# ### Test the Complete Pipeline

# %%
# ============================================================================
# TEST THE COMPLETE PIPELINE
# ============================================================================

logger.section("🧪 TESTING COMPLETE PIPELINE")

# Test cases covering various issues
test_cases = [
    # Diacritics
    "الأُمَمُ المُتَّحِدَة مُنَظَّمَة دَوْلِيَّة.",
    
    # Mixed punctuation
    "أولاً, ثانياً; ثالثاً?",
    
    # Alef variations
    "أحمد وإبراهيم وآدم",
    
    # Numbers
    "في عام 2024 وصل العدد إلى 100",
    
    # Latin text
    "الوثيقة A/47/10 والقرار UNDP/2024",
    
    # Consecutive punctuation
    "ماذا؟؟؟ هذا صحيح...",
    
    # Whitespace issues
    "النص   مع    مسافات   كثيرة",
    
    # Short sentence (should be filtered)
    "نعم",
    
    # Combined issues
    "وَقَالَ أحمد: هذا مُهِمٌّ جِدَّاً,, والله!!",
]

logger.info("Processing test cases:\n")

for i, test in enumerate(test_cases, 1):
    result = preprocessor.preprocess_line(test, apply_optional=False)
    logger.info(f"Test {i}:")
    logger.info(f"   Input:  '{test}'")
    if result:
        logger.info(f"   Output: '{result}'")
    else:
        logger.info(f"   Output: [FILTERED]")
    logger.info("")

# %% [markdown]
# ---
# ## 6. Part 5: Post-Preprocessing Inspection

# %%
# ============================================================================
# SECTION 6: POST-PREPROCESSING INSPECTION
# ============================================================================

def inspect_preprocessed_data(file_path: str, sample_size: int = 100000) -> Dict:
    """
    Inspect preprocessed data to verify quality.
    
    Parameters:
    -----------
    file_path : str
        Path to preprocessed file
    sample_size : int
        Number of lines to inspect
        
    Returns:
    --------
    Dict
        Inspection results
    """
    logger.section("🔍 POST-PREPROCESSING INSPECTION")
    logger.info(f"Inspecting: {file_path}")
    
    stats = {
        'total_lines': 0,
        'total_words': 0,
        'total_chars': 0,
        'word_counts': [],
        'remaining_issues': {
            'diacritics': 0,
            'latin_letters': 0,
            'western_numbers': 0,
            'latin_punct': 0,
            'consecutive_punct': 0,
            'alef_variations': 0,
        },
        'punctuation_dist': Counter(),
        'sample_lines': [],
    }
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return stats
    
    # Patterns for checking
    diacritics_pattern = re.compile(r'[\u064B-\u0652]')
    latin_pattern = re.compile(r'[A-Za-z]')
    western_num_pattern = re.compile(r'[0-9]')
    consecutive_punct_pattern = re.compile(r'[،؛؟.,:;?!]{2,}')
    alef_var_pattern = re.compile(r'[أإآٱ]')
    arabic_word_pattern = re.compile(r'[\u0600-\u06FF]+')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            line = line.rstrip('\n')
            stats['total_lines'] += 1
            stats['total_chars'] += len(line)
            
            words = arabic_word_pattern.findall(line)
            stats['total_words'] += len(words)
            stats['word_counts'].append(len(words))
            
            # Store sample lines
            if len(stats['sample_lines']) < 10:
                stats['sample_lines'].append(line)
            
            # Check for remaining issues
            if diacritics_pattern.search(line):
                stats['remaining_issues']['diacritics'] += 1
            if latin_pattern.search(line):
                stats['remaining_issues']['latin_letters'] += 1
            if western_num_pattern.search(line):
                stats['remaining_issues']['western_numbers'] += 1
            if consecutive_punct_pattern.search(line):
                stats['remaining_issues']['consecutive_punct'] += 1
            if alef_var_pattern.search(line):
                stats['remaining_issues']['alef_variations'] += 1
            
            # Count punctuation
            for char in line:
                if char in VALID_PUNCTUATION:
                    stats['punctuation_dist'][char] += 1
    
    # Display results
    logger.subsection("Basic Statistics")
    logger.info(f"Total lines: {stats['total_lines']:,}")
    logger.info(f"Total words: {stats['total_words']:,}")
    logger.info(f"Total chars: {stats['total_chars']:,}")
    
    if stats['word_counts']:
        word_arr = np.array(stats['word_counts'])
        logger.info(f"Avg words/line: {np.mean(word_arr):.2f}")
        logger.info(f"Min words: {np.min(word_arr)}")
        logger.info(f"Max words: {np.max(word_arr)}")
    
    logger.subsection("Remaining Issues Check")
    total_issues = sum(stats['remaining_issues'].values())
    if total_issues == 0:
        logger.success("No issues found! Data is clean.")
    else:
        logger.warn(f"Found {total_issues} potential issues:")
        for issue, count in stats['remaining_issues'].items():
            if count > 0:
                logger.info(f"   {issue}: {count:,} lines")
    
    logger.subsection("Punctuation Distribution")
    for char, count in stats['punctuation_dist'].most_common():
        name = ARABIC_PUNCTUATION.get(char, 'Unknown')
        logger.info(f"   '{char}' ({name}): {count:,}")
    
    logger.subsection("Sample Lines")
    for i, line in enumerate(stats['sample_lines'][:5], 1):
        display = line[:80] + "..." if len(line) > 80 else line
        logger.info(f"   {i}. {display}")
    
    return stats


# This will be run after processing the dataset

# %% [markdown]
# ---
# ## 7. Part 6: Save Preprocessed Data

# %%
# ============================================================================
# SECTION 7: PROCESS AND SAVE DATASET
# ============================================================================

def run_preprocessing_pipeline(
    input_dir: str,
    output_dir: str,
    config: PreprocessingConfig,
    create_variants: bool = True,
    sample_size: Optional[int] = None
):
    """
    Run the complete preprocessing pipeline and save results.
    
    Creates multiple output variants:
    1. mandatory_only.txt - Only mandatory preprocessing
    2. with_waw_separation.txt - + waw separation
    3. with_number_tokens.txt - + number replacement
    4. full_preprocessing.txt - All preprocessing steps
    
    Parameters:
    -----------
    input_dir : str
        Path to input dataset
    output_dir : str
        Path to output directory
    config : PreprocessingConfig
        Configuration object
    create_variants : bool
        Whether to create multiple preprocessing variants
    sample_size : Optional[int]
        Limit processing (None = full dataset)
    """
    logger.section("🚀 RUNNING PREPROCESSING PIPELINE")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ArabicTextPreprocessor(config)
    
    # ============================================
    # Variant 1: Mandatory preprocessing only
    # ============================================
    logger.subsection("Variant 1: Mandatory Preprocessing Only")
    
    output_file_mandatory = os.path.join(output_dir, "mandatory_only.txt")
    
    # Ensure optional steps are off
    config.separate_waw_conjunction = False
    config.replace_numbers_with_token = False
    config.remove_foreign_terms = False
    
    preprocessor = ArabicTextPreprocessor(config)
    stats_mandatory = preprocessor.process_dataset(
        input_dir, 
        output_file_mandatory,
        apply_optional=False,
        sample_size=sample_size
    )
    
    # Save stats
    stats_file = os.path.join(output_dir, "stats_mandatory.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_input': stats_mandatory.total_input_lines,
            'total_output': stats_mandatory.total_output_lines,
            'empty_removed': stats_mandatory.empty_lines_removed,
            'short_removed': stats_mandatory.short_lines_removed,
            'long_truncated': stats_mandatory.long_lines_truncated,
        }, f, indent=2)
    
    if not create_variants:
        return
    
    # ============================================
    # Variant 2: With Waw Separation
    # ============================================
    logger.subsection("Variant 2: With Waw Separation")
    
    config.separate_waw_conjunction = True
    preprocessor = ArabicTextPreprocessor(config)
    
    output_file_waw = os.path.join(output_dir, "with_waw_separation.txt")
    stats_waw = preprocessor.process_dataset(
        input_dir,
        output_file_waw,
        apply_optional=True,
        sample_size=sample_size
    )
    
    config.separate_waw_conjunction = False  # Reset
    
    # ============================================
    # Variant 3: With Number Tokens
    # ============================================
    logger.subsection("Variant 3: With Number Tokens")
    
    config.replace_numbers_with_token = True
    preprocessor = ArabicTextPreprocessor(config)
    
    output_file_nums = os.path.join(output_dir, "with_number_tokens.txt")
    stats_nums = preprocessor.process_dataset(
        input_dir,
        output_file_nums,
        apply_optional=True,
        sample_size=sample_size
    )
    
    config.replace_numbers_with_token = False  # Reset
    
    # ============================================
    # Variant 4: Full Preprocessing
    # ============================================
    logger.subsection("Variant 4: Full Preprocessing (All Options)")
    
    config.separate_waw_conjunction = True
    config.replace_numbers_with_token = True
    config.remove_foreign_terms = True
    
    preprocessor = ArabicTextPreprocessor(config)
    
    output_file_full = os.path.join(output_dir, "full_preprocessing.txt")
    stats_full = preprocessor.process_dataset(
        input_dir,
        output_file_full,
        apply_optional=True,
        sample_size=sample_size
    )
    
    logger.section("✅ PREPROCESSING COMPLETE")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("Files created:")
    logger.info(f"   1. mandatory_only.txt")
    logger.info(f"   2. with_waw_separation.txt")
    logger.info(f"   3. with_number_tokens.txt")
    logger.info(f"   4. full_preprocessing.txt")


# %%
# ============================================================================
# EXECUTE PREPROCESSING
# ============================================================================

# Configuration for execution
EXECUTE_FULL_PIPELINE = True  # Set to True to run on full dataset
SAMPLE_SIZE_FOR_TEST = 100000  # Set to None for full dataset

if EXECUTE_FULL_PIPELINE:
    logger.section("⚡ EXECUTING PREPROCESSING PIPELINE")
    
    # For testing, use a sample
    sample = SAMPLE_SIZE_FOR_TEST  # Change to None for full processing
    
    run_preprocessing_pipeline(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        config=config,
        create_variants=True,
        sample_size=sample
    )
    
    # Inspect the mandatory preprocessing output
    mandatory_file = os.path.join(config.output_dir, "mandatory_only.txt")
    if os.path.exists(mandatory_file):
        post_stats = inspect_preprocessed_data(mandatory_file, sample_size=50000)
else:
    logger.info("Pipeline execution skipped. Set EXECUTE_FULL_PIPELINE = True to run.")

# %% [markdown]
# ---
# ## 8. Summary & Recommendations

# %%
# ============================================================================
# SECTION 8: SUMMARY AND RECOMMENDATIONS
# ============================================================================

logger.section("📋 PREPROCESSING SUMMARY & RECOMMENDATIONS")

summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PREPROCESSING PIPELINE SUMMARY                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MANDATORY PREPROCESSING STEPS (Always Applied):                             ║
║  ─────────────────────────────────────────────────                           ║
║  ✅ Remove diacritics (tashkeel)                                             ║
║  ✅ Normalize Alef variations (أإآٱ → ا)                                      ║
║  ✅ Normalize Alef Maksura (ى → ي)                                           ║
║  ✅ Remove Tatweel (ـ)                                                       ║
║  ✅ Unify punctuation to Arabic (,;? → ،؛؟)                                  ║
║  ✅ Unify numbers to Arabic (0-9 → ٠-٩)                                      ║
║  ✅ Remove Latin letters                                                     ║
║  ✅ Remove OOV characters                                                    ║
║  ✅ Handle consecutive punctuation                                           ║
║  ✅ Normalize whitespace                                                     ║
║  ✅ Add punctuation spacing                                                  ║
║  ✅ Filter empty lines                                                       ║
║  ✅ Filter short sentences (<3 words)                                        ║
║  ✅ Truncate long sentences (>100 words)                                     ║
║                                                                              ║
║  OPTIONAL PREPROCESSING STEPS (For Experimentation):                         ║
║  ─────────────────────────────────────────────────────                       ║
║  ⚙️  Separate Waw conjunction (و + word → و word)                            ║
║  ⚙️  Replace numbers with token (<NUM>)                                      ║
║  ⚙️  Remove document references (A/47/10)                                    ║
║  ⚙️  Remove/mark stopwords                                                   ║
║  ⚙️  Replace rare words (<UNK>)                                              ║
║                                                                              ║
║  OUTPUT FILES CREATED:                                                       ║
║  ─────────────────────                                                       ║
║  📁 preprocessed_data/                                                       ║
║     ├── mandatory_only.txt      (Recommended for most experiments)          ║
║     ├── with_waw_separation.txt (Test waw separation effect)                ║
║     ├── with_number_tokens.txt  (Test number normalization effect)          ║
║     └── full_preprocessing.txt  (All preprocessing applied)                 ║
║                                                                              ║
║  RECOMMENDATIONS:                                                            ║
║  ────────────────                                                            ║
║  1. Start with mandatory_only.txt for baseline experiments                   ║
║  2. Use with_waw_separation.txt if conjunction handling helps                ║
║  3. Test with_number_tokens.txt if numbers cause vocabulary issues          ║
║  4. Compare results to determine optimal preprocessing                       ║
║                                                                              ║
║  NEXT STEPS:                                                                 ║
║  ───────────                                                                 ║
║  1. Tokenize preprocessed data with chosen tokenizer                        ║
║  2. Create train/validation/test splits                                     ║
║  3. Generate labels for sequence-to-sequence task                           ║
║  4. Train and evaluate models                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print(summary)

# %%
# ============================================================================
# LABEL GENERATION UTILITY
# ============================================================================

def generate_labels_for_line(text: str) -> Tuple[List[str], List[int]]:
    """
    Generate word and label sequences for a preprocessed line.
    
    This is the format needed for sequence-to-sequence training.
    
    Parameters:
    -----------
    text : str
        Preprocessed text line
        
    Returns:
    --------
    Tuple[List[str], List[int]]
        (words, labels) where labels indicate punctuation after each word
        
    Label Mapping:
    - 0: No punctuation (O)
    - 1: Period (.)
    - 2: Arabic Comma (،)
    - 3: Question Mark (؟)
    - 4: Semicolon (؛)
    - 5: Colon (:)
    - 6: Exclamation (!)
    """
    LABEL_MAP = {
        'O': 0,    # No punctuation
        '.': 1,    # Period
        '،': 2,    # Arabic Comma
        '؟': 3,    # Question Mark
        '؛': 4,    # Semicolon
        ':': 5,    # Colon
        '!': 6,    # Exclamation
    }
    
    words = []
    labels = []
    
    # Pattern for Arabic words
    word_pattern = re.compile(r'[\u0600-\u06FF٠-٩]+')
    
    # Find all words and their positions
    for match in word_pattern.finditer(text):
        word = match.group()
        end_pos = match.end()
        
        # Look for punctuation immediately after
        remaining = text[end_pos:].lstrip()
        
        if remaining and remaining[0] in LABEL_MAP:
            label = LABEL_MAP[remaining[0]]
        else:
            label = LABEL_MAP['O']
        
        words.append(word)
        labels.append(label)
    
    return words, labels


# Test label generation
logger.section("🏷️ LABEL GENERATION EXAMPLE")

test_lines = [
    "هذا نص عربي، يحتوي على علامات ترقيم.",
    "ما هو السؤال؟",
    "أولاً؛ ثانياً؛ ثالثاً.",
]

logger.info("Label Mapping:")
logger.info("   0: No punctuation (O)")
logger.info("   1: Period (.)")
logger.info("   2: Comma (،)")
logger.info("   3: Question (؟)")
logger.info("   4: Semicolon (؛)")
logger.info("   5: Colon (:)")
logger.info("   6: Exclamation (!)")
logger.info("")

for text in test_lines:
    words, labels = generate_labels_for_line(text)
    logger.info(f"Text: {text}")
    logger.info(f"Words:  {words}")
    logger.info(f"Labels: {labels}")
    logger.info("")

# %%
logger.section("✅ PREPROCESSING NOTEBOOK COMPLETE")
logger.info("All preprocessing functions and pipeline have been defined.")
logger.info("Run the pipeline with EXECUTE_FULL_PIPELINE = True to process the full dataset.")