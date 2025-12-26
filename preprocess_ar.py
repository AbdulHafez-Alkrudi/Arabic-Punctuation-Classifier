# %% [markdown]
# # 🔧 خط أنابيب المعالجة المسبقة لمجموعة بيانات الترقيم العربي
# ## معالجة مكون SSAC-UNPC
# 
# ---
# 
# ### 📋 جدول المحتويات
# 
# 1. [المقدمة والإعداد](#1-المقدمة-والإعداد)
# 2. [الجزء 1: فحص المشكلات (قبل المعالجة)](#2-الجزء-1-فحص-المشكلات-قبل-المعالجة)
#    - 2.1 مشكلات مستوى الحروف
#    - 2.2 مشكلات علامات الترقيم
#    - 2.3 مشكلات مستوى الجمل
#    - 2.4 مشكلات الأنماط الخاصة
# 3. [الجزء 2: خطوات المعالجة الإلزامية](#3-الجزء-2-خطوات-المعالجة-الإلزامية)
#    - 3.1 إزالة التشكيل (الحركات)
#    - 3.2 توحيد أشكال الألف
#    - 3.3 توحيد التاء المربوطة والألف المقصورة
#    - 3.4 إزالة الحروف خارج النطاق (OOV)
#    - 3.5 إزالة الحروف اللاتينية
#    - 3.6 توحيد الأرقام (الأرقام العربية)
#    - 3.7 توحيد علامات الترقيم (الترقيم العربي)
#    - 3.8 معالجة الترقيم المتتابع
#    - 3.9 توحيد المسافات وضبط مسافات الترقيم
#    - 3.10 إزالة الأسطر الفارغة والقصيرة جداً
#    - 3.11 معالجة الجمل الطويلة
# 4. [الجزء 3: خطوات المعالجة الاختيارية (للتجريب)](#4-الجزء-3-خطوات-المعالجة-الاختيارية)
#    - 4.1 فصل واو العطف عن الكلمات
#    - 4.2 استراتيجيات التعامل مع كلمات التوقف
#    - 4.3 استبدال الأرقام برمز خاص
#    - 4.4 معالجة الكلمات النادرة
#    - 4.5 تسوية طول الجمل
#    - 4.6 إزالة/استبدال المصطلحات الأجنبية
# 5. [الجزء 4: خط أنابيب المعالجة الكامل](#5-الجزء-4-خط-أنابيب-المعالجة-الكامل)
# 6. [الجزء 5: الفحص بعد المعالجة](#6-الجزء-5-الفحص-بعد-المعالجة)
# 7. [الجزء 6: حفظ البيانات المعالجة](#7-الجزء-6-حفظ-البيانات-المعالجة)
# 8. [الملخص والتوصيات](#8-الملخص-والتوصيات)
# 
# ---

# %% [markdown]
# ## 1. المقدمة والإعداد
# 
# ### 🎯 الغرض من هذا الدفتر
# 
# ينفذ هذا الدفتر خط أنابيب معالجة مسبقة شامل لمكون SSAC-UNPC 
# من مجموعة بيانات الترقيم العربي (APD). بناءً على نتائج تحليل البيانات الاستكشافي (EDA)، نعالج:
# 
# **المشكلات التي تم تحديدها في التحليل:**
# 
# | المشكلة | الشدة | الحل |
# |-------|----------|----------|
# | وجود التشكيل (0.27%) | متوسطة | إزالة جميع الحركات |
# | خليط ترقيم عربي/لاتيني | عالية | التوحيد إلى العربي |
# | خليط أرقام عربية/غربية | متوسطة | التوحيد إلى العربي |
# | أشكال متعددة للألف | منخفضة | التوحيد إلى ألف مجردة |
# | حروف خارج النطاق | منخفضة | إزالة أو استبدال |
# | حروف لاتينية في النص | متوسطة | إزالة |
# | جمل قصيرة جداً (<3 كلمات) | متوسطة | تصفية (حذف) |
# | جمل طويلة جداً (>100 كلمة) | منخفضة | قص أو تقسيم |
# | ترقيم متتابع | منخفضة | معالجة مناسبة |
# | ترقيم ملتصق بالكلمات | متوسطة | إضافة مسافات |
# 
# ### 📊 النتائج المتوقعة
# 
# بعد المعالجة:
# - نص عربي نظيف ومتسق
# - نظام ترقيم موحد (عربي فقط)
# - نظام أرقام موحد (عربي فقط)
# - خلو من التشكيل والعلامات الخاصة
# - مسافات صحيحة حول علامات الترقيم
# - تصفية الجمل القصيرة/الإشكالية
# - جاهزية للترميز (Tokenization) وتدريب النموذج

# %%
# ============================================================================
# القسم 1: الاستيراد والإعداد
# ============================================================================

# -----------------------------
# مكتبات بايثون القياسية
# -----------------------------
import os                      # لعمليات النظام (المسارات)
import re                      # للتعابير النمطية (Regex)
import sys                     # لوظائف النظام
import json                    # للتعامل مع ملفات JSON
import random                  # للتوليد العشوائي
from pathlib import Path       # للتعامل الحديث مع المسارات
from collections import Counter # للعد الفعال
from typing import List, Tuple, Dict, Optional, Generator, Callable # لأنماط البيانات
from dataclasses import dataclass, field # لتعريف هياكل البيانات

# -----------------------------
# مكتبات تحليل البيانات
# -----------------------------
import numpy as np             # للعمليات الحسابية

# -----------------------------
# شريط التقدم
# -----------------------------
try:
    from tqdm import tqdm      # شريط تقدم للعمليات الطويلة
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("ملاحظة: مكتبة tqdm غير مثبتة. للتثبيت: pip install tqdm")

# %%
# ============================================================================
# القسم 2: إعداد المسجل (LOGGER)
# ============================================================================

class NotebookLogger:
    """
    مسجل موحد ومبسط لدفاتر جوبيتر.
    
    - يطبع المخرجات في الدفتر
    - يضيف السجلات إلى ملف
    - بدون طوابع زمنية
    - بدون ترويسات الجلسة
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
        self._write(f"⚠️  تحذير: {message}\n")

    def error(self, message: str):
        self._write(f"❌ خطأ: {message}\n")

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


# تهيئة المسجل
logger = NotebookLogger(log_file="logs/preprocessing.log")

# %%
# ============================================================================
# القسم 3: الإعدادات (CONFIGURATION)
# ============================================================================

@dataclass
class PreprocessingConfig:
    """
    إعدادات خط أنابيب المعالجة المسبقة.
    
    يفصل بين خطوات المعالجة الإلزامية والاختيارية.
    """
    
    # -----------------------------
    # مسارات الملفات
    # -----------------------------
    input_dir: str = "../SSAC-UNPC"
    output_dir: str = "preprocessed_data"
    log_dir: str = "logs"
    
    # -----------------------------
    # المعالجة الإلزامية (تطبق دائماً)
    # -----------------------------
    remove_diacritics: bool = True       # إزالة التشكيل
    normalize_alef: bool = True          # توحيد الألف
    normalize_teh_marbuta: bool = True   # ة (اختياري، البعض يحولها لـ ه)
    normalize_alef_maksura: bool = True  # ى -> ي
    remove_tatweel: bool = True          # إزالة التطويل (ـ)
    remove_latin_letters: bool = True    # إزالة الحروف اللاتينية
    remove_oov_chars: bool = True        # إزالة الحروف خارج النطاق
    unify_numbers_to_arabic: bool = True # توحيد الأرقام للعربية
    unify_punctuation_to_arabic: bool = True # توحيد الترقيم للعربي
    handle_consecutive_punct: bool = True # معالجة الترقيم المتتابع
    normalize_whitespace: bool = True    # توحيد المسافات
    add_punct_spacing: bool = True       # إضافة مسافات للترقيم
    
    # -----------------------------
    # تصفية الجمل
    # -----------------------------
    min_words: int = 3
    max_words: int = 100
    remove_empty_lines: bool = True
    
    # -----------------------------
    # المعالجة الاختيارية (للتجريب)
    # -----------------------------
    separate_waw_conjunction: bool = False   # فصل واو العطف
    remove_stopwords: bool = False           # إزالة كلمات التوقف
    replace_numbers_with_token: bool = False # استبدال الأرقام برمز
    replace_rare_words: bool = False         # استبدال الكلمات النادرة
    rare_word_threshold: int = 5             # حد الكلمات النادرة
    remove_foreign_terms: bool = False       # إزالة المصطلحات الأجنبية
    
    # -----------------------------
    # معاملات المعالجة
    # -----------------------------
    sample_size: Optional[int] = None  # None = معالجة الكل
    chunk_size: int = 100000           # عدد الأسطر لكل دفعة (للكفاءة)
    random_seed: int = 42


# تهيئة الإعدادات
config = PreprocessingConfig()

# إنشاء مجلدات المخرجات
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

logger.info("✅ تم تهيئة الإعدادات!")
logger.info(f"   مجلد المدخلات: {config.input_dir}")
logger.info(f"   مجلد المخرجات: {config.output_dir}")

# %%
# ============================================================================
# القسم 4: تعريفات الحروف العربية
# ============================================================================

# -----------------------------
# الحروف العربية الصالحة
# -----------------------------
# بناءً على نتائج التحليل: جميع الحروف العربية الموجودة في مجموعة البيانات

ARABIC_LETTERS = set(
    'ء آ أ ؤ إ ئ ا ب ة ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي ى'
    .split()
)

# مجموعة موسعة تشمل الحروف الأقل شيوعاً (تأثير فارسي/أردو في الأسماء)
ARABIC_LETTERS_EXTENDED = ARABIC_LETTERS | set('پ چ ژ گ ڤ')

# -----------------------------
# التشكيل العربي (الحركات)
# -----------------------------
ARABIC_DIACRITICS = {
    '\u064B': 'فتحتان',     # ً
    '\u064C': 'ضمتان',      # ٌ
    '\u064D': 'كسرتان',     # ٍ
    '\u064E': 'فتحة',       # َ
    '\u064F': 'ضمة',        # ُ
    '\u0650': 'كسرة',       # ِ
    '\u0651': 'شدة',        # ّ
    '\u0652': 'سكون',       # ْ
}

DIACRITICS_PATTERN = re.compile(r'[\u064B-\u0652]')

# -----------------------------
# الأرقام العربية
# -----------------------------
ARABIC_NUMERALS = '٠١٢٣٤٥٦٧٨٩'
WESTERN_NUMERALS = '0123456789'

# جداول التحويل
WESTERN_TO_ARABIC_NUMS = str.maketrans(WESTERN_NUMERALS, ARABIC_NUMERALS)
ARABIC_TO_WESTERN_NUMS = str.maketrans(ARABIC_NUMERALS, WESTERN_NUMERALS)

# -----------------------------
# علامات الترقيم
# -----------------------------
# الترقيم المستهدف (العربي)
ARABIC_PUNCTUATION = {
    '،': 'فاصلة عربية',
    '؛': 'فاصلة منقوطة عربية',
    '؟': 'علامة استفهام عربية',
    '.': 'نقطة',
    ':': 'نقطتان رأسيتان',
    '!': 'علامة تعجب',
}

# المعادلات اللاتينية للتوحيد
LATIN_TO_ARABIC_PUNCT = {
    ',': '،',   # فاصلة لاتينية -> عربية
    ';': '؛',   # فاصلة منقوطة لاتينية -> عربية
    '?': '؟',   # علامة استفهام لاتينية -> عربية
}

# جميع علامات الترقيم الصالحة (بعد التوحيد)
VALID_PUNCTUATION = set(ARABIC_PUNCTUATION.keys())

# علامات نهاية الجملة
SENTENCE_TERMINALS = {'.', '؟', '!'}

# -----------------------------
# أشكال الألف
# -----------------------------
ALEF_VARIATIONS = {
    'أ': 'ا',  # ألف همزة فوق
    'إ': 'ا',  # ألف همزة تحت
    'آ': 'ا',  # ألف مدة
    'ٱ': 'ا',  # ألف وصل
}

# -----------------------------
# كلمات التوقف العربية
# -----------------------------
ARABIC_STOPWORDS = set([
    # حروف الجر
    'في', 'من', 'على', 'إلى', 'الى', 'عن', 'مع', 'بين', 'عند', 'حتى', 'منذ',
    'الي', 'فى', 'علي',
    # أسماء الإشارة
    'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك',
    # الأسماء الموصولة
    'التي', 'الذي', 'اللذان', 'اللتان', 'الذين', 'اللاتي', 'اللواتي',
    # أدوات العطف
    'و', 'أو', 'او', 'ثم', 'لكن', 'بل', 'إذا', 'لو', 'إذ', 'ف',
    # أدوات أخرى
    'أن', 'ان', 'إن', 'قد', 'لا', 'ما', 'لم', 'لن', 'ل', 'ب', 'ك',
    # الضمائر
    'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم', 'انا', 'انت',
    # الأفعال المساعدة
    'كان', 'كانت', 'يكون', 'تكون', 'كانوا', 'ليس', 'ليست',
    # أخرى
    'كل', 'بعض', 'أي', 'اي', 'غير', 'بعد', 'قبل', 'حيث', 'عندما',
    'حول', 'دون', 'ضد', 'خلال', 'عبر', 'نحو', 'فوق', 'تحت',
    # كلمات وظيفية شائعة
    'وفي', 'ومن', 'وعلى', 'والى', 'ومع', 'وهو', 'وهي', 'وهذا', 'وهذه',
    'فإن', 'فان', 'وإن', 'وان', 'لأن', 'لان', 'كما', 'مما', 'إنه', 'انه',
    'إنها', 'انها', 'أنه', 'أنها', 'ذات', 'لها', 'له', 'لهم', 'بها', 'به',
    'فيها', 'فيه', 'منها', 'منه', 'عنها', 'عنه', 'إليها', 'إليه',
    'عليها', 'عليه', 'معها', 'معه', 'بينها', 'بينهم',
])

# -----------------------------
# أنماط واو العطف
# -----------------------------
# الحد الأدنى لطول الكلمة لفصل الواو
WAW_CONJUNCTION_MIN_LENGTH = 3  # افصل فقط إذا تبقى 3 حروف أو أكثر

logger.info("✅ تم تحميل تعريفات الحروف العربية!")
logger.info(f"   - الحروف العربية: {len(ARABIC_LETTERS)}")
logger.info(f"   - التشكيل: {len(ARABIC_DIACRITICS)}")
logger.info(f"   - كلمات التوقف: {len(ARABIC_STOPWORDS)}")
logger.info(f"   - علامات الترقيم الصالحة: {len(VALID_PUNCTUATION)}")

# %%
# ============================================================================
# القسم 5: أدوات تحميل البيانات
# ============================================================================

def iter_dataset_lines(dataset_dir: str, encoding: str = "utf-8") -> Generator[str, None, None]:
    """
    تكرار (Iterate) على جميع ملفات مجموعة البيانات كتدفق أسطر واحد.
    
    تطبق هذه الدالة التحميل الكسول (Lazy Loading) للتعامل مع البيانات الكبيرة
    التي لا يمكن تحميلها في الذاكرة دفعة واحدة.
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار المجلد المحتوي على ملفات .txt
    encoding : str
        ترميز الملف (الافتراضي: utf-8)
        
    تنتج (Yields):
    -------
    str
        جملة/سطر واحد في كل مرة (بدون محرف السطر الجديد)
    """
    # الحصول على جميع ملفات النصوص مرتبة
    txt_files = sorted(Path(dataset_dir).glob("*.txt"))
    
    if not txt_files:
        logger.error(f"لم يتم العثور على ملفات .txt في {dataset_dir}")
        return
    
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line in f:
                    yield line.rstrip("\n")
        except Exception as e:
            logger.error(f"خطأ في قراءة {file_path}: {e}")
            continue


def count_total_lines(dataset_dir: str) -> int:
    """
    عد إجمالي الأسطر في مجموعة البيانات (لشريط التقدم).
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار مجلد البيانات
        
    ترجع:
    --------
    int
        إجمالي عدد الأسطر
    """
    total = 0
    for file_path in sorted(Path(dataset_dir).glob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def get_sample_lines(dataset_dir: str, n: int = 1000, seed: int = 42) -> List[str]:
    """
    الحصول على عينة عشوائية من الأسطر للفحص.
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار مجلد البيانات
    n : int
        عدد العينات
    seed : int
        البذرة العشوائية
        
    ترجع:
    --------
    List[str]
        قائمة أسطر العينة
    """
    random.seed(seed)
    
    # جمع الأسطر من البداية لأخذ العينات
    lines = []
    for i, line in enumerate(iter_dataset_lines(dataset_dir)):
        if i >= n * 10:  # جمع أكثر من المطلوب للاختيار العشوائي
            break
        if line.strip():
            lines.append(line)
    
    return random.sample(lines, min(n, len(lines)))


logger.info("✅ أدوات تحميل البيانات جاهزة!")

# %% [markdown]
# ---
# ## 2. الجزء 1: فحص المشكلات (قبل المعالجة)
# 
# قبل تطبيق أي معالجة، نحتاج لتحديد وقياس جميع المشكلات في البيانات الخام بشكل منهجي.
# يساعدنا هذا في:
# 
# 1. فهم حجم كل مشكلة
# 2. ترتيب أولويات خطوات المعالجة
# 3. التحقق من أن المعالجة أصلحت المشكلات

# %% [markdown]
# ### 2.1 مشكلات مستوى الحروف

# %%
# ============================================================================
# الفحص 2.1: مشكلات مستوى الحروف
# ============================================================================

def inspect_character_issues(dataset_dir: str, sample_size: int = 500000) -> Dict:
    """
    فحص مشكلات مستوى الحروف في مجموعة البيانات.
    
    يتحقق من:
    - التشكيل (الحركات)
    - الحروف خارج النطاق (OOV)
    - الحروف اللاتينية
    - الحروف الخاصة
    - أشكال الألف
    - التطويل (ـ)
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار مجلد البيانات
    sample_size : int
        عدد الأسطر للفحص
        
    ترجع:
    --------
    Dict
        قاموس يحتوي على إحصائيات المشكلات
    """
    logger.section("🔍 فحص مشكلات مستوى الحروف")
    logger.info(f"جاري فحص {sample_size:,} سطر...")
    
    # تهيئة العدادات
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
    
    # تعريف مجموعة الحروف الصالحة
    valid_chars = set()
    valid_chars.update(ARABIC_LETTERS_EXTENDED)
    valid_chars.update(ARABIC_NUMERALS)
    valid_chars.update(WESTERN_NUMERALS)
    valid_chars.update(VALID_PUNCTUATION)
    valid_chars.update(LATIN_TO_ARABIC_PUNCT.keys())
    valid_chars.update(' \t\n')  # المسافات
    valid_chars.update('()[]{}«»""\'-–—')  # الأقواس وعلامات التنصيص
    valid_chars.update(ARABIC_DIACRITICS.keys())  # التشكيل (للعد)
    
    # نمط الحروف اللاتينية
    latin_pattern = re.compile(r'[A-Za-z]')
    
    # إنشاء المكرر (Iterator)
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="فحص الحروف")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        stats['total_chars'] += len(line)
        
        has_diacritics = False
        has_latin = False
        has_oov = False
        
        for char in line:
            # التحقق من التشكيل
            if char in ARABIC_DIACRITICS:
                stats['diacritics'][char] += 1
                has_diacritics = True
            
            # التحقق من الحروف اللاتينية
            if latin_pattern.match(char):
                stats['latin_letters'][char] += 1
                has_latin = True
            
            # التحقق من أشكال الألف
            if char in ALEF_VARIATIONS:
                stats['alef_variations'][char] += 1
            
            # التحقق من التطويل
            if char == '\u0640':
                stats['tatweel_count'] += 1
            
            # التحقق من الحروف خارج النطاق
            if char not in valid_chars and not latin_pattern.match(char):
                stats['oov_chars'][char] += 1
                has_oov = True
        
        if has_diacritics:
            stats['lines_with_diacritics'] += 1
        if has_latin:
            stats['lines_with_latin'] += 1
        if has_oov:
            stats['lines_with_oov'] += 1
    
    # عرض النتائج
    logger.subsection("التشكيل (الحركات)")
    total_diacritics = sum(stats['diacritics'].values())
    logger.info(f"إجمالي التشكيل الموجود: {total_diacritics:,}")
    logger.info(f"أسطر تحتوي على تشكيل: {stats['lines_with_diacritics']:,} ({stats['lines_with_diacritics']/stats['total_lines']*100:.2f}%)")
    
    if stats['diacritics']:
        logger.info("توزيع التشكيل:")
        for char, count in stats['diacritics'].most_common():
            name = ARABIC_DIACRITICS.get(char, 'غير معروف')
            logger.info(f"   {repr(char)} ({name}): {count:,}")
    
    logger.subsection("الحروف اللاتينية")
    total_latin = sum(stats['latin_letters'].values())
    logger.info(f"إجمالي الحروف اللاتينية: {total_latin:,}")
    logger.info(f"أسطر تحتوي على لاتيني: {stats['lines_with_latin']:,} ({stats['lines_with_latin']/stats['total_lines']*100:.2f}%)")
    
    if stats['latin_letters']:
        logger.info("أكثر الحروف اللاتينية:")
        for char, count in stats['latin_letters'].most_common(10):
            logger.info(f"   '{char}': {count:,}")
    
    logger.subsection("أشكال الألف")
    total_alef_var = sum(stats['alef_variations'].values())
    logger.info(f"إجمالي أشكال الألف: {total_alef_var:,}")
    
    if stats['alef_variations']:
        for char, count in stats['alef_variations'].most_common():
            logger.info(f"   '{char}': {count:,}")
    
    logger.subsection("التطويل (الكشيدة)")
    logger.info(f"عدد مرات التطويل: {stats['tatweel_count']:,}")
    
    logger.subsection("حروف خارج النطاق (OOV)")
    total_oov = sum(stats['oov_chars'].values())
    logger.info(f"إجمالي الحروف خارج النطاق: {total_oov:,}")
    logger.info(f"أسطر تحتوي على OOV: {stats['lines_with_oov']:,} ({stats['lines_with_oov']/stats['total_lines']*100:.2f}%)")
    
    if stats['oov_chars']:
        logger.info("أهم الحروف خارج النطاق:")
        for char, count in stats['oov_chars'].most_common(20):
            try:
                char_name = f"U+{ord(char):04X}"
            except:
                char_name = "Unknown"
            logger.info(f"   {repr(char)} ({char_name}): {count:,}")
    
    return stats


# تشغيل فحص الحروف
char_issues = inspect_character_issues(config.input_dir, sample_size=500000)

# %% [markdown]
# ### 2.2 مشكلات علامات الترقيم

# %%
# ============================================================================
# الفحص 2.2: مشكلات علامات الترقيم
# ============================================================================

def inspect_punctuation_issues(dataset_dir: str, sample_size: int = 500000) -> Dict:
    """
    فحص المشكلات المتعلقة بعلامات الترقيم.
    
    يتحقق من:
    - خليط الترقيم العربي/اللاتيني
    - الترقيم المتتابع (مثل ،،، أو ..)
    - نقص المسافات حول الترقيم
    - علامات ترقيم غير صالحة
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار مجلد البيانات
    sample_size : int
        عدد الأسطر للفحص
        
    ترجع:
    --------
    Dict
        قاموس يحتوي على إحصائيات المشكلات
    """
    logger.section("🔍 فحص مشكلات الترقيم")
    logger.info(f"جاري فحص {sample_size:,} سطر...")
    
    stats = {
        'total_lines': 0,
        'arabic_punct': Counter(),
        'latin_punct': Counter(),
        'other_punct': Counter(),
        'consecutive_punct': [],  # لتخزين أمثلة
        'consecutive_punct_count': 0,
        'attached_punct_count': 0,
        'lines_with_mixed_punct': 0,
    }
    
    # جميع علامات الترقيم للكشف
    all_punct = set(ARABIC_PUNCTUATION.keys()) | set(LATIN_TO_ARABIC_PUNCT.keys())
    
    # نمط الترقيم المتتابع
    consecutive_pattern = re.compile(r'[،؛؟.,:;?!]{2,}')
    
    # نمط الترقيم الملتصق (بدون مسافة قبله/بعده)
    # كلمة عربية متبوعة مباشرة بترقيم بدون مسافة
    attached_pattern = re.compile(r'[\u0600-\u06FF][،؛؟.:!][\u0600-\u06FF]')
    
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="فحص الترقيم")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        
        has_arabic_punct = False
        has_latin_punct = False
        
        # عد أنواع الترقيم
        for char in line:
            if char in ARABIC_PUNCTUATION:
                stats['arabic_punct'][char] += 1
                has_arabic_punct = True
            elif char in LATIN_TO_ARABIC_PUNCT:
                stats['latin_punct'][char] += 1
                has_latin_punct = True
            elif char in '()[]{}«»""\'':
                stats['other_punct'][char] += 1
        
        # التحقق من الترقيم المختلط
        if has_arabic_punct and has_latin_punct:
            stats['lines_with_mixed_punct'] += 1
        
        # التحقق من الترقيم المتتابع
        consecutive_matches = consecutive_pattern.findall(line)
        if consecutive_matches:
            stats['consecutive_punct_count'] += len(consecutive_matches)
            if len(stats['consecutive_punct']) < 20:  # تخزين أمثلة
                for match in consecutive_matches:
                    stats['consecutive_punct'].append((match, line[:100]))
        
        # التحقق من الترقيم الملتصق
        attached_matches = attached_pattern.findall(line)
        if attached_matches:
            stats['attached_punct_count'] += len(attached_matches)
    
    # عرض النتائج
    logger.subsection("الترقيم العربي")
    for char, count in stats['arabic_punct'].most_common():
        name = ARABIC_PUNCTUATION.get(char, 'غير معروف')
        logger.info(f"   '{char}' ({name}): {count:,}")
    
    logger.subsection("الترقيم اللاتيني (يحتاج توحيد)")
    for char, count in stats['latin_punct'].most_common():
        logger.info(f"   '{char}': {count:,} -> يجب أن يصبح '{LATIN_TO_ARABIC_PUNCT.get(char, char)}'")
    
    logger.subsection("أسطر بترقيم مختلط")
    logger.info(f"أسطر تحوي خليطاً من الترقيم العربي واللاتيني: {stats['lines_with_mixed_punct']:,}")
    logger.info(f"النسبة: {stats['lines_with_mixed_punct']/stats['total_lines']*100:.2f}%")
    
    logger.subsection("الترقيم المتتابع")
    logger.info(f"إجمالي الحالات: {stats['consecutive_punct_count']:,}")
    if stats['consecutive_punct']:
        logger.info("أمثلة:")
        for match, context in stats['consecutive_punct'][:5]:
            logger.info(f"   '{match}' في: {context[:60]}...")
    
    logger.subsection("الترقيم الملتصق")
    logger.info(f"حالات يفتقد فيها الترقيم لمسافات مناسبة: {stats['attached_punct_count']:,}")
    
    return stats


# تشغيل فحص الترقيم
punct_issues = inspect_punctuation_issues(config.input_dir, sample_size=500000)

# %% [markdown]
# ### 2.3 مشكلات مستوى الجمل

# %%
# ============================================================================
# الفحص 2.3: مشكلات مستوى الجمل
# ============================================================================

def inspect_sentence_issues(dataset_dir: str, sample_size: int = 1000000) -> Dict:
    """
    فحص مشكلات مستوى الجمل.
    
    يتحقق من:
    - الأسطر الفارغة
    - الجمل القصيرة جداً
    - الجمل الطويلة جداً
    - غياب علامات نهاية الجملة
    - جمل متعددة في سطر واحد
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار مجلد البيانات
    sample_size : int
        عدد الأسطر للفحص
        
    ترجع:
    --------
    Dict
        قاموس يحتوي على إحصائيات المشكلات
    """
    logger.section("🔍 فحص مشكلات مستوى الجمل")
    logger.info(f"جاري فحص {sample_size:,} سطر...")
    
    stats = {
        'total_lines': 0,
        'empty_lines': 0,
        'very_short': [],  # < 3 كلمات
        'very_long': [],   # > 100 كلمة
        'word_counts': [],
        'missing_terminal': 0,
        'wrong_terminal': Counter(),
        'multiple_terminals': 0,
    }
    
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="فحص الجمل")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        
        # التحقق من الفراغ
        stripped = line.strip()
        if not stripped:
            stats['empty_lines'] += 1
            continue
        
        # عد الكلمات
        words = stripped.split()
        word_count = len(words)
        stats['word_counts'].append(word_count)
        
        # التحقق من القصر الشديد
        if word_count < 3:
            if len(stats['very_short']) < 20:  # تخزين أمثلة
                stats['very_short'].append(stripped)
        
        # التحقق من الطول الشديد
        if word_count > 100:
            if len(stats['very_long']) < 10:
                stats['very_long'].append((word_count, stripped[:100] + "..."))
        
        # التحقق من علامة النهاية
        if stripped:
            last_char = stripped[-1]
            if last_char not in SENTENCE_TERMINALS:
                stats['missing_terminal'] += 1
                stats['wrong_terminal'][last_char] += 1
        
        # التحقق من وجود علامات نهاية متعددة داخل السطر
        terminal_count = sum(1 for c in stripped[:-1] if c in SENTENCE_TERMINALS)
        if terminal_count > 0:
            stats['multiple_terminals'] += 1
    
    # عرض النتائج
    logger.subsection("الأسطر الفارغة")
    logger.info(f"أسطر فارغة: {stats['empty_lines']:,} ({stats['empty_lines']/stats['total_lines']*100:.4f}%)")
    
    logger.subsection("إحصائيات طول الجمل")
    if stats['word_counts']:
        word_arr = np.array(stats['word_counts'])
        logger.info(f"المتوسط: {np.mean(word_arr):.2f} كلمة")
        logger.info(f"الوسيط: {np.median(word_arr):.2f} كلمة")
        logger.info(f"الأدنى: {np.min(word_arr)} كلمات")
        logger.info(f"الأقصى: {np.max(word_arr)} كلمات")
        logger.info(f"الانحراف المعياري: {np.std(word_arr):.2f} كلمة")
    
    logger.subsection("الجمل القصيرة جداً (<3 كلمات)")
    short_count = sum(1 for w in stats['word_counts'] if w < 3)
    logger.info(f"العدد: {short_count:,} ({short_count/stats['total_lines']*100:.2f}%)")
    if stats['very_short']:
        logger.info("أمثلة:")
        for example in stats['very_short'][:5]:
            logger.info(f"   '{example}'")
    
    logger.subsection("الجمل الطويلة جداً (>100 كلمة)")
    long_count = sum(1 for w in stats['word_counts'] if w > 100)
    logger.info(f"العدد: {long_count:,} ({long_count/stats['total_lines']*100:.2f}%)")
    if stats['very_long']:
        logger.info("أمثلة:")
        for wc, example in stats['very_long'][:3]:
            logger.info(f"   [{wc} كلمة] {example}")
    
    logger.subsection("مشكلات علامات النهاية")
    logger.info(f"أسطر تفتقد لعلامة نهاية قياسية: {stats['missing_terminal']:,}")
    if stats['wrong_terminal']:
        logger.info("نهايات غير قياسية وجدت:")
        for char, count in stats['wrong_terminal'].most_common(10):
            logger.info(f"   '{char}': {count:,}")
    
    logger.subsection("علامات نهاية متعددة")
    logger.info(f"أسطر تحتوي على علامات نهاية في وسط الجملة: {stats['multiple_terminals']:,}")
    
    return stats


# تشغيل فحص الجمل
sentence_issues = inspect_sentence_issues(config.input_dir, sample_size=1000000)

# %% [markdown]
# ### 2.4 مشكلات الأنماط الخاصة

# %%
# ============================================================================
# الفحص 2.4: مشكلات الأنماط الخاصة
# ============================================================================

def inspect_special_patterns(dataset_dir: str, sample_size: int = 500000) -> Dict:
    """
    فحص الأنماط الخاصة التي قد تحتاج لمعالجة.
    
    يتحقق من:
    - واو العطف المتصلة بالكلمات
    - الأرقام (العربية والغربية)
    - مراجع المستندات (مثل A/47/10)
    - أنماط الروابط أو البريد الإلكتروني
    - الحروف المكررة
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار مجلد البيانات
    sample_size : int
        عدد الأسطر للفحص
        
    ترجع:
    --------
    Dict
        قاموس يحتوي على إحصائيات الأنماط
    """
    logger.section("🔍 فحص الأنماط الخاصة")
    logger.info(f"جاري فحص {sample_size:,} سطر...")
    
    stats = {
        'total_lines': 0,
        'waw_attached': Counter(),  # كلمات تبدأ بـ و
        'arabic_numbers': 0,
        'western_numbers': 0,
        'mixed_number_lines': 0,
        'doc_references': [],
        'repeated_chars': [],
        'foreign_words': Counter(),
    }
    
    # الأنماط (Patterns)
    waw_word_pattern = re.compile(r'\bو[\u0600-\u06FF]{2,}\b')
    arabic_num_pattern = re.compile(r'[٠-٩]+')
    western_num_pattern = re.compile(r'[0-9]+')
    doc_ref_pattern = re.compile(r'[A-Z]/[0-9]+(?:/[0-9A-Z]+)*')
    repeated_pattern = re.compile(r'(.)\1{3,}')  # نفس الحرف 4 مرات أو أكثر
    foreign_word_pattern = re.compile(r'\b[A-Za-z]{3,}\b')  # 3+ حروف لاتينية
    
    iterator = iter_dataset_lines(dataset_dir)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=sample_size, desc="فحص الأنماط")
    
    for i, line in enumerate(iterator):
        if i >= sample_size:
            break
        
        stats['total_lines'] += 1
        
        # الكلمات المتصلة بالواو
        waw_matches = waw_word_pattern.findall(line)
        for match in waw_matches:
            stats['waw_attached'][match] += 1
        
        # أنظمة الأرقام
        has_arabic = bool(arabic_num_pattern.search(line))
        has_western = bool(western_num_pattern.search(line))
        
        if has_arabic:
            stats['arabic_numbers'] += len(arabic_num_pattern.findall(line))
        if has_western:
            stats['western_numbers'] += len(western_num_pattern.findall(line))
        if has_arabic and has_western:
            stats['mixed_number_lines'] += 1
        
        # مراجع المستندات
        doc_refs = doc_ref_pattern.findall(line)
        if doc_refs and len(stats['doc_references']) < 20:
            stats['doc_references'].extend(doc_refs[:2])
        
        # الحروف المكررة
        repeated = repeated_pattern.findall(line)
        if repeated and len(stats['repeated_chars']) < 10:
            stats['repeated_chars'].append(line[:80])
        
        # الكلمات الأجنبية
        foreign = foreign_word_pattern.findall(line)
        for word in foreign:
            stats['foreign_words'][word] += 1
    
    # عرض النتائج
    logger.subsection("أنماط واو العطف")
    logger.info(f"إجمالي الكلمات التي تبدأ بـ و: {sum(stats['waw_attached'].values()):,}")
    logger.info("أكثر الكلمات شيوعاً مع الواو:")
    for word, count in stats['waw_attached'].most_common(15):
        logger.info(f"   '{word}': {count:,}")
    
    logger.subsection("أنظمة الأرقام")
    logger.info(f"أرقام عربية وجدت: {stats['arabic_numbers']:,}")
    logger.info(f"أرقام غربية وجدت: {stats['western_numbers']:,}")
    logger.info(f"أسطر بأرقام مختلطة: {stats['mixed_number_lines']:,}")
    
    logger.subsection("مراجع المستندات")
    logger.info(f"أمثلة: {stats['doc_references'][:10]}")
    
    logger.subsection("حروف مكررة")
    if stats['repeated_chars']:
        logger.info("أسطر بحروف مكررة:")
        for example in stats['repeated_chars'][:3]:
            logger.info(f"   {example}")
    else:
        logger.info("لم يتم العثور على أنماط حروف مكررة مهمة")
    
    logger.subsection("كلمات أجنبية")
    logger.info(f"كلمات أجنبية فريدة: {len(stats['foreign_words']):,}")
    logger.info("أكثر الكلمات الأجنبية شيوعاً:")
    for word, count in stats['foreign_words'].most_common(15):
        logger.info(f"   '{word}': {count:,}")
    
    return stats


# تشغيل فحص الأنماط الخاصة
special_issues = inspect_special_patterns(config.input_dir, sample_size=500000)

# %% [markdown]
# ---
# ## 3. الجزء 2: خطوات المعالجة الإلزامية
# 
# هذه الخطوات **تُطبق دائماً** لأنها تعالج مشكلات أساسية في جودة البيانات
# والتي ستؤثر سلباً على تدريب النموذج.

# %% [markdown]
# ### 3.1 إزالة التشكيل (الحركات)

# %%
# ============================================================================
# المعالجة 3.1: إزالة التشكيل
# ============================================================================

def remove_diacritics(text: str) -> str:
    """
    إزالة التشكيل (الحركات) من النص العربي.
    
    التشكيل الذي يتم إزالته:
    - فتحتان (ً)، ضمتان (ٌ)، كسرتان (ٍ)
    - فتحة (َ)، ضمة (ُ)، كسرة (ِ)
    - شدة (ّ)، سكون (ْ)
    
    المعاملات:
    -----------
    text : str
        النص المدخل مع احتمال وجود تشكيل
        
    ترجع:
    --------
    str
        النص بدون تشكيل
        
    مثال:
    --------
    >>> remove_diacritics("الْعَرَبِيَّة")
    'العربية'
    """
    return DIACRITICS_PATTERN.sub('', text)


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: إزالة التشكيل")

test_cases = [
    "الْعَرَبِيَّة",
    "مُحَمَّد",
    "الأُمَمُ المُتَّحِدَة",
    "نص بدون تشكيل",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result = remove_diacritics(test)
    logger.info(f"   '{test}' -> '{result}'")

# %% [markdown]
# ### 3.2 توحيد أشكال الألف

# %%
# ============================================================================
# المعالجة 3.2: توحيد أشكال الألف
# ============================================================================

# تجميع النمط للكفاءة
ALEF_PATTERN = re.compile(r'[أإآٱ]')

def normalize_alef(text: str) -> str:
    """
    توحيد جميع أشكال الألف إلى ألف مجردة (ا).
    
    التحويلات:
    - أ (همزة فوق) -> ا
    - إ (همزة تحت) -> ا
    - آ (مدة) -> ا
    - ٱ (وصل) -> ا
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص مع ألف موحدة
        
    مثال:
    --------
    >>> normalize_alef("أحمد إبراهيم آدم")
    'احمد ابراهيم ادم'
    """
    return ALEF_PATTERN.sub('ا', text)


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: توحيد الألف")

test_cases = [
    "أحمد",
    "إبراهيم",
    "آدم",
    "الأمم",
    "الإنسان",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result = normalize_alef(test)
    logger.info(f"   '{test}' -> '{result}'")

# %% [markdown]
# ### 3.3 توحيد التاء المربوطة والألف المقصورة

# %%
# ============================================================================
# المعالجة 3.3: توحيد التاء المربوطة والألف المقصورة
# ============================================================================

def normalize_teh_marbuta(text: str, to_heh: bool = False) -> str:
    """
    معالجة التاء المربوطة (ة).
    
    خيارات:
    - إبقاؤها كما هي (الافتراضي لهذه المهمة)
    - تحويلها إلى هاء (ه) - بعض تطبيقات NLP تفعل ذلك
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    to_heh : bool
        إذا كان True، حول ة إلى ه
        
    ترجع:
    --------
    str
        النص المعالج
    """
    if to_heh:
        return text.replace('ة', 'ه')
    return text


def normalize_alef_maksura(text: str) -> str:
    """
    توحيد الألف المقصورة (ى) إلى ياء (ي).
    
    ملاحظة: في بعض السياقات، يتم التمييز بينهما. لتنبؤ الترقيم،
    التوحيد يحسن الاتساق.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص مع ى -> ي
        
    مثال:
    --------
    >>> normalize_alef_maksura("على")
    'علي'
    """
    return text.replace('ى', 'ي')


# اختبار الدوال
logger.section("🔧 المعالجة المسبقة: توحيد التاء المربوطة والألف المقصورة")

test_cases = [
    ("مدرسة", "تاء مربوطة"),
    ("على", "ألف مقصورة"),
    ("مستشفى", "ألف مقصورة"),
    ("القاهرة", "تاء مربوطة"),
]

logger.info("حالات الاختبار:")
for test, note in test_cases:
    result_tm = normalize_teh_marbuta(test, to_heh=False)
    result_am = normalize_alef_maksura(test)
    logger.info(f"   '{test}' ({note})")
    logger.info(f"      تاء مربوطة (إبقاء): '{result_tm}'")
    logger.info(f"      ألف مقصورة -> ياء: '{result_am}'")

# %% [markdown]
# ### 3.4 إزالة الحروف خارج النطاق (OOV)

# %%
# ============================================================================
# المعالجة 3.4: إزالة الحروف خارج النطاق (OOV)
# ============================================================================

def build_valid_charset() -> set:
    """
    بناء مجموعة الحروف الصالحة لمهمة الترقيم العربي.
    
    تشمل:
    - الحروف العربية (الأساسية + الموسعة)
    - الأرقام العربية
    - علامات الترقيم الصالحة
    - المسافات
    
    ترجع:
    --------
    set
        مجموعة الحروف الصالحة
    """
    valid = set()
    
    # الحروف العربية (أساسية + موسعة)
    valid.update('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوي')
    valid.update('ىپچژگڤ')  # موسعة
    
    # الأرقام العربية
    valid.update('٠١٢٣٤٥٦٧٨٩')
    
    # علامات الترقيم الصالحة
    valid.update('،؛؟.:!')
    
    # حروف هيكلية أساسية
    valid.update(' ')  # مسافة
    
    # إبقاء بعض الأقواس للهيكلة (ستعالج لاحقاً إذا لزم)
    valid.update('()[]')
    
    return valid


VALID_CHARSET = build_valid_charset()


def remove_oov_characters(text: str, valid_chars: set = None, replacement: str = '') -> str:
    """
    إزالة الحروف غير الموجودة في مجموعة الحروف الصالحة.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    valid_chars : set
        مجموعة الحروف الصالحة (يستخدم VALID_CHARSET إذا None)
    replacement : str
        الحرف البديل للحروف المحذوفة (الافتراضي: إزالة)
        
    ترجع:
    --------
    str
        النص بعد إزالة الحروف غير الصالحة
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


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: إزالة الحروف خارج النطاق")

test_cases = [
    "النص العربي مع English text",
    "رقم: ١٢٣ و 456",
    "مع رموز خاصة: @#$%",
    "«نص بين أقواس»",
]

logger.info(f"حجم مجموعة الحروف الصالحة: {len(VALID_CHARSET)}")
logger.info("حالات الاختبار:")
for test in test_cases:
    result = remove_oov_characters(test)
    logger.info(f"   '{test}'")
    logger.info(f"   -> '{result}'")

# %% [markdown]
# ### 3.5 إزالة الحروف اللاتينية

# %%
# ============================================================================
# المعالجة 3.5: إزالة الحروف اللاتينية
# ============================================================================

LATIN_PATTERN = re.compile(r'[A-Za-z]+')

def remove_latin_letters(text: str, replacement: str = '') -> str:
    """
    إزالة الحروف اللاتينية من النص.
    
    يعالج:
    - الكلمات الإنجليزية المفردة
    - مراجع المستندات (A/47/10 -> /47/10)
    - النص المختلط
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    replacement : str
        بديل للتتابعات اللاتينية (الافتراضي: إزالة)
        
    ترجع:
    --------
    str
        النص بدون حروف لاتينية
    """
    return LATIN_PATTERN.sub(replacement, text)


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: إزالة الحروف اللاتينية")

test_cases = [
    "الأمم المتحدة United Nations",
    "الوثيقة A/47/10",
    "برنامج UNDP للتنمية",
    "add. 1",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result = remove_latin_letters(test)
    logger.info(f"   '{test}' -> '{result}'")

# %% [markdown]
# ### 3.6 توحيد الأرقام (الأرقام العربية)

# %%
# ============================================================================
# المعالجة 3.6: توحيد الأرقام للعربية
# ============================================================================

def unify_numbers_to_arabic(text: str) -> str:
    """
    تحويل جميع الأرقام الغربية (0-9) إلى أرقام عربية (٠-٩).
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بأرقام عربية فقط
        
    مثال:
    --------
    >>> unify_numbers_to_arabic("عام 2024")
    'عام ٢٠٢٤'
    """
    return text.translate(WESTERN_TO_ARABIC_NUMS)


def unify_numbers_to_western(text: str) -> str:
    """
    تحويل جميع الأرقام العربية (٠-٩) إلى أرقام غربية (0-9).
    
    نهج بديل - بعض النماذج تفضل الأرقام الغربية.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بأرقام غربية فقط
    """
    return text.translate(ARABIC_TO_WESTERN_NUMS)


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: توحيد الأرقام")

test_cases = [
    "عام 2024",
    "رقم ١٢٣",
    "مبلغ 100 دولار",
    "٥٠٠ + 500 = ١٠٠٠",
]

logger.info("حالات الاختبار (-> أرقام عربية):")
for test in test_cases:
    result = unify_numbers_to_arabic(test)
    logger.info(f"   '{test}' -> '{result}'")

# %% [markdown]
# ### 3.7 توحيد علامات الترقيم (الترقيم العربي)

# %%
# ============================================================================
# المعالجة 3.7: توحيد الترقيم للعربي
# ============================================================================

def unify_punctuation_to_arabic(text: str) -> str:
    """
    تحويل علامات الترقيم اللاتينية إلى نظيراتها العربية.
    
    التحويلات:
    - , -> ،  (فاصلة)
    - ; -> ؛  (فاصلة منقوطة)
    - ? -> ؟  (علامة استفهام)
    
    ملاحظة: النقطة (.)، النقطتان (:)، والتعجب (!) تبقى كما هي
    لأنها مشتركة في النظامين.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بترقيم عربي
    """
    for latin, arabic in LATIN_TO_ARABIC_PUNCT.items():
        text = text.replace(latin, arabic)
    return text


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: توحيد الترقيم")

test_cases = [
    "أولاً, ثانياً, ثالثاً",
    "هل هذا صحيح?",
    "ملاحظة; هذا مهم",
    "النص يحتوي على , و ; و ?",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result = unify_punctuation_to_arabic(test)
    logger.info(f"   '{test}'")
    logger.info(f"   -> '{result}'")

# %% [markdown]
# ### 3.8 معالجة الترقيم المتتابع

# %%
# ============================================================================
# المعالجة 3.8: معالجة الترقيم المتتابع
# ============================================================================

def handle_consecutive_punctuation(text: str, keep_first: bool = True) -> str:
    """
    معالجة علامات الترقيم المتتابعة.
    
    الاستراتيجيات:
    - keep_first: إبقاء العلامة الأولى وحذف الباقي
    - keep_last: إبقاء العلامة الأخيرة وحذف الباقي
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    keep_first : bool
        إذا True، احتفظ بالأولى
        
    ترجع:
    --------
    str
        النص بعد معالجة التتابع
    """
    # نمط يطابق 2 أو أكثر من علامات الترقيم المتتابعة
    punct_chars = r'،؛؟.,:;?!'
    pattern = re.compile(f'([{punct_chars}])([{punct_chars}]+)')
    
    if keep_first:
        # إبقاء الأولى
        return pattern.sub(r'\1', text)
    else:
        # إبقاء الأخيرة
        def keep_last_match(m):
            return m.group(0)[-1]
        return pattern.sub(keep_last_match, text)


def handle_consecutive_punctuation_smart(text: str) -> str:
    """
    معالجة ذكية للترقيم المتتابع.
    
    تستخدم الأولوية: . > ؟ > ! > ؛ > ، > :
    تحتفظ بالعلامة ذات الأولوية الأعلى.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بعد اختزال الترقيم
    """
    # ترتيب الأولوية (الأعلى أولاً)
    priority = {'.': 6, '؟': 5, '?': 5, '!': 4, '؛': 3, ';': 3, '،': 2, ',': 2, ':': 1}
    
    punct_chars = r'،؛؟.,:;?!'
    pattern = re.compile(f'[{punct_chars}]{{2,}}')
    
    def replace_func(match):
        sequence = match.group(0)
        # إيجاد العلامة ذات الأولوية الأعلى
        best_char = sequence[0]
        best_priority = priority.get(best_char, 0)
        
        for char in sequence:
            char_priority = priority.get(char, 0)
            if char_priority > best_priority:
                best_char = char
                best_priority = char_priority
        
        return best_char
    
    return pattern.sub(replace_func, text)


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: معالجة الترقيم المتتابع")

test_cases = [
    "ماذا؟؟؟",
    "هذا صحيح..",
    "أولاً،،",
    "انتهى.؟",
    "نهاية النص.,",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result_first = handle_consecutive_punctuation(test, keep_first=True)
    result_smart = handle_consecutive_punctuation_smart(test)
    logger.info(f"   '{test}'")
    logger.info(f"      إبقاء الأولى: '{result_first}'")
    logger.info(f"      ذكي:          '{result_smart}'")

# %% [markdown]
# ### 3.9 توحيد المسافات وضبط مسافات الترقيم

# %%
# ============================================================================
# المعالجة 3.9: توحيد المسافات وضبط مسافات الترقيم
# ============================================================================

def normalize_whitespace(text: str) -> str:
    """
    توحيد المسافات في النص.
    
    - استبدال المسافات المتعددة بمسافة واحدة
    - إزالة المسافات في البداية والنهاية
    - استبدال علامات الجدولة (tabs) بمسافات
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بمسافات موحدة
    """
    # استبدال tabs بمسافات
    text = text.replace('\t', ' ')
    
    # استبدال المسافات المتعددة
    text = re.sub(r' +', ' ', text)
    
    # إزالة مسافات البداية والنهاية
    text = text.strip()
    
    return text


def add_punctuation_spacing(text: str) -> str:
    """
    ضمان وجود مسافات صحيحة حول علامات الترقيم.
    
    القواعد:
    - مسافة بعد علامة الترقيم (إذا تبعها حرف/رقم)
    - لا مسافة قبل علامة الترقيم
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بمسافات ترقيم صحيحة
    """
    punct_marks = '،؛؟.:!'
    
    # إزالة المسافة قبل الترقيم
    for p in punct_marks:
        text = re.sub(rf'\s+{re.escape(p)}', p, text)
    
    # إضافة مسافة بعد الترقيم إذا تبعه حرف أو رقم عربي
    for p in punct_marks:
        # بعد الترقيم، إذا تبعه حرف عربي بدون مسافة، أضف مسافة
        text = re.sub(
            rf'{re.escape(p)}([\u0600-\u06FF٠-٩])',
            rf'{p} \1',
            text
        )
    
    # تنظيف المسافات المتعددة التي قد تكون نشأت
    text = re.sub(r' +', ' ', text)
    
    return text


# اختبار الدوال
logger.section("🔧 المعالجة المسبقة: توحيد المسافات وضبط الترقيم")

test_cases = [
    "النص   مع   مسافات    كثيرة",
    "كلمة،كلمة",
    "نص .مع مسافة قبل النقطة",
    "سؤال؟جواب",
    "أولاً ، ثانياً ، ثالثاً",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result_ws = normalize_whitespace(test)
    result_spacing = add_punctuation_spacing(result_ws)
    logger.info(f"   '{test}'")
    logger.info(f"      بعد توحيد المسافات: '{result_ws}'")
    logger.info(f"      بعد ضبط الترقيم:    '{result_spacing}'")

# %% [markdown]
# ### 3.10 إزالة الأسطر الفارغة والقصيرة جداً

# %%
# ============================================================================
# المعالجة 3.10: إزالة الأسطر الفارغة والقصيرة
# ============================================================================

def is_valid_sentence(text: str, min_words: int = 3) -> bool:
    """
    التحقق مما إذا كانت الجملة تستوفي الحد الأدنى من المتطلبات.
    
    المعاملات:
    -----------
    text : str
        النص المدخل (يجب أن يكون معالجاً مسبقاً)
    min_words : int
        الحد الأدنى للكلمات
        
    ترجع:
    --------
    bool
        True إذا كانت الجملة صالحة
    """
    # إزالة المسافات
    text = text.strip()
    
    # التحقق من الفراغ
    if not text:
        return False
    
    # عد الكلمات العربية فقط
    arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
    
    return len(arabic_words) >= min_words


def filter_sentence(text: str, min_words: int = 3) -> Optional[str]:
    """
    تصفية وإرجاع الجملة إذا كانت صالحة، وإلا None.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    min_words : int
        الحد الأدنى للكلمات
        
    ترجع:
    --------
    Optional[str]
        الجملة إذا كانت صالحة، وإلا None
    """
    if is_valid_sentence(text, min_words):
        return text
    return None


# اختبار الدالة
logger.section("🔧 المعالجة المسبقة: تصفية الجمل القصيرة/الفارغة")

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

logger.info(f"الحد الأدنى للكلمات: {config.min_words}")
logger.info("حالات الاختبار:")
for test in test_cases:
    is_valid = is_valid_sentence(test, min_words=config.min_words)
    status = "✅ صالح" if is_valid else "❌ غير صالح"
    logger.info(f"   '{test}' -> {status}")

# %% [markdown]
# ### 3.11 معالجة الجمل الطويلة

# %%
# ============================================================================
# المعالجة 3.11: معالجة الجمل الطويلة
# ============================================================================

def truncate_sentence(text: str, max_words: int = 100) -> str:
    """
    قص الجملة لتناسب الحد الأقصى للكلمات.
    
    الاستراتيجية: القص عند حدود الكلمات، محاولة الإنهاء بعلامة ترقيم.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    max_words : int
        الحد الأقصى للكلمات
        
    ترجع:
    --------
    str
        النص المقصوص
    """
    words = text.split()
    
    if len(words) <= max_words:
        return text
    
    # القص إلى max_words
    truncated_words = words[:max_words]
    truncated = ' '.join(truncated_words)
    
    # ضمان النهاية بعلامة ترقيم
    if truncated and truncated[-1] not in SENTENCE_TERMINALS:
        truncated += '.'
    
    return truncated


def split_long_sentence(text: str, max_words: int = 100) -> List[str]:
    """
    تقسيم الجملة الطويلة إلى جمل متعددة عند الحدود الطبيعية.
    
    الاستراتيجية: التقسيم عند علامات الترقيم (،؛) التي تخلق فواصل طبيعية.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    max_words : int
        الحد الأقصى للكلمات لكل مقطع
        
    ترجع:
    --------
    List[str]
        قائمة بمقاطع الجملة
    """
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    
    # إيجاد نقاط الفصل المحتملة (بعد ، أو ؛)
    segments = []
    current_segment = []
    word_count = 0
    
    for word in words:
        current_segment.append(word)
        word_count += 1
        
        # تحقق إذا كانت الكلمة تنتهي بفاصلة وتجاوزنا المنتصف
        if word_count >= max_words // 2:
            if word.endswith('،') or word.endswith('؛'):
                # إنشاء مقطع
                segment_text = ' '.join(current_segment)
                segments.append(segment_text)
                current_segment = []
                word_count = 0
        
        # إجبار التقسيم إذا وصلنا للحد الأقصى
        if word_count >= max_words:
            segment_text = ' '.join(current_segment)
            if not segment_text.endswith(('.', '؟', '!')):
                segment_text += '.'
            segments.append(segment_text)
            current_segment = []
            word_count = 0
    
    # إضافة المتبقي
    if current_segment:
        segment_text = ' '.join(current_segment)
        segments.append(segment_text)
    
    return segments


# اختبار الدوال
logger.section("🔧 المعالجة المسبقة: معالجة الجمل الطويلة")

# إنشاء جملة طويلة للاختبار
long_sentence = " ".join(["كلمة"] * 150)
logger.info(f"الطول الأصلي: {len(long_sentence.split())} كلمة")

truncated = truncate_sentence(long_sentence, max_words=100)
logger.info(f"بعد القص: {len(truncated.split())} كلمة")
logger.info(f"ينتهي بـ: '{truncated[-10:]}'")

# اختبار الفواصل الطبيعية
long_with_punct = "هذا نص طويل يحتوي على فقرات متعددة، وكل فقرة تحتوي على معلومات مهمة، " * 20
segments = split_long_sentence(long_with_punct, max_words=50)
logger.info(f"\nتم التقسيم إلى {len(segments)} مقطع:")
for i, seg in enumerate(segments[:3]):
    logger.info(f"   المقطع {i+1}: {len(seg.split())} كلمة")

# %% [markdown]
# ---
# ## 4. الجزء 3: خطوات المعالجة الاختيارية (للتجريب)
# 
# هذه الخطوات **اختيارية** ويمكن تفعيلها للتجريب.
# قد تحسن أو تضر بأداء النموذج حسب المهمة.

# %% [markdown]
# ### 4.1 فصل واو العطف عن الكلمات

# %%
# ============================================================================
# اختياري 4.1: فصل واو العطف
# ============================================================================

def separate_waw_conjunction(text: str, min_remaining_length: int = 3) -> str:
    """
    فصل واو العطف (و) عن بداية الكلمات.
    
    في العربية، تتصل الواو بالكلمة التالية كسابقة. فصلها قد يساعد في:
    - ترميز (Tokenization) أفضل
    - حدود كلمات متسقة
    - تحسين تنبؤ الترقيم قبل العطف
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    min_remaining_length : int
        افصل فقط إذا تبقى من الكلمة هذا العدد من الحروف
        
    ترجع:
    --------
    str
        النص مع واو عطف مفصولة
        
    مثال:
    --------
    >>> separate_waw_conjunction("وقال الرجل وذهب")
    'و قال الرجل و ذهب'
    """
    # كلمات حيث الواو جزء من الجذر (يجب عدم فصلها)
    waw_root_words = {
        'وقت', 'وجه', 'وضع', 'وصل', 'وقع', 'وزن', 'وفد', 'ورق', 'وطن',
        'وسط', 'وحدة', 'وزير', 'وزارة', 'ولاية', 'ولد', 'والد', 'والدة',
        'وثيقة', 'وثائق', 'واقع', 'واجب', 'وفاة', 'وكالة', 'وكيل',
        'واحد', 'واحدة', 'وسيلة', 'وسائل', 'ورشة', 'وظيفة', 'وظائف',
    }
    
    def should_separate(word: str) -> bool:
        """تحقق مما إذا كان يجب فصل الواو عن هذه الكلمة."""
        if not word.startswith('و'):
            return False
        
        if len(word) < min_remaining_length + 1:  # +1 للواو
            return False
        
        # التحقق مما إذا كانت الكلمة (مع الواو) كلمة جذرية
        if word in waw_root_words:
            return False
        
        # التحقق مما إذا كانت الكلمة بدون واو موجودة ككلمة صالحة
        remaining = word[1:]
        
        # بادئات شائعة تشير إلى أن الواو للعطف
        conjunction_indicators = [
            'ال',   # و + ال (التعريف)
            'هو', 'هي', 'هم',  # ضمائر
            'قد', 'لم', 'لن',  # أدوات
            'كان', 'يكون',     # أفعال
            'أن', 'إن',        # أدوات
        ]
        
        for indicator in conjunction_indicators:
            if remaining.startswith(indicator):
                return True
        
        # إذا بدأت الكلمة المتبقية بـ ال التعريف، فهي غالباً عطف
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


# اختبار الدالة
logger.section("🔧 اختياري: فصل واو العطف")

test_cases = [
    "وقال الرجل",
    "والأمم المتحدة",
    "وفي هذا الصدد",
    "وقت الاجتماع",  # لا يجب فصلها (وقت كلمة جذرية)
    "وثيقة مهمة",   # لا يجب فصلها
    "والتنمية المستدامة",
    "وهو يعمل",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result = separate_waw_conjunction(test)
    logger.info(f"   '{test}'")
    logger.info(f"   -> '{result}'")

# %% [markdown]
# ### 4.2 استراتيجيات التعامل مع كلمات التوقف

# %%
# ============================================================================
# اختياري 4.2: التعامل مع كلمات التوقف
# ============================================================================

def mark_stopwords(text: str, marker: str = '<SW>') -> str:
    """
    تمييز كلمات التوقف برمز خاص (للتحليل/التجريب).
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    marker : str
        الرمز للإضافة بعد كلمات التوقف
        
    ترجع:
    --------
    str
        النص مع كلمات توقف مميزة
    """
    words = text.split()
    result = []
    
    for word in words:
        # إزالة الترقيم للفحص
        clean_word = re.sub(r'[،؛؟.:!]', '', word)
        
        if clean_word in ARABIC_STOPWORDS:
            result.append(word + marker)
        else:
            result.append(word)
    
    return ' '.join(result)


def remove_stopwords(text: str, keep_structure: bool = True) -> str:
    """
    إزالة كلمات التوقف من النص.
    
    تحذير: هذا قد يضر بتنبؤ الترقيم لأن كلمات التوقف توفر سياقاً هيكلياً مهماً.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    keep_structure : bool
        إذا True، احتفظ بالترقيم حتى لو كان ملتصقاً بكلمات توقف
        
    ترجع:
    --------
    str
        النص بدون كلمات توقف
    """
    words = text.split()
    result = []
    
    for word in words:
        # فصل الكلمة عن الترقيم اللاحق
        punct = ''
        clean_word = word
        
        if word and word[-1] in '،؛؟.:!':
            punct = word[-1]
            clean_word = word[:-1]
        
        if clean_word not in ARABIC_STOPWORDS:
            result.append(word)
        elif keep_structure and punct:
            # الاحتفاظ بالترقيم حتى لو كانت الكلمة كلمة توقف
            if result:
                result[-1] += punct
    
    return ' '.join(result)


# اختبار الدالة
logger.section("🔧 اختياري: التعامل مع كلمات التوقف")

test_cases = [
    "وفي هذا الصدد",
    "من أجل التنمية",
    "الأمم المتحدة هي منظمة دولية",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    marked = mark_stopwords(test)
    removed = remove_stopwords(test)
    logger.info(f"   الأصل:   '{test}'")
    logger.info(f"   مميزة:   '{marked}'")
    logger.info(f"   محذوفة:  '{removed}'")

# %% [markdown]
# ### 4.3 استبدال الأرقام برمز خاص

# %%
# ============================================================================
# اختياري 4.3: استبدال الأرقام برمز
# ============================================================================

def replace_numbers_with_token(text: str, token: str = '<NUM>') -> str:
    """
    استبدال جميع الأرقام برمز خاص.
    
    هذا يساعد في:
    - تقليل حجم المفردات
    - تركيز النموذج على الهيكل بدلاً من أرقام محددة
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    token : str
        الرمز لاستبدال الأرقام به
        
    ترجع:
    --------
    str
        النص بعد استبدال الأرقام
    """
    # نمط للأرقام العربية أو الغربية
    number_pattern = re.compile(r'[٠-٩0-9]+')
    return number_pattern.sub(token, text)


def normalize_number_format(text: str) -> str:
    """
    توحيد تنسيق الأرقام (مثل فواصل الآلاف).
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بتنسيق أرقام موحد
    """
    # إزالة فواصل الآلاف (كل من , و ٬)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'([\u0660-\u0669])٬([\u0660-\u0669])', r'\1\2', text)
    
    return text


# اختبار الدالة
logger.section("🔧 اختياري: استبدال الأرقام برمز")

test_cases = [
    "في عام ٢٠٢٤",
    "مبلغ ١٠٠٠٠٠ دولار",
    "من ٥ إلى ١٠ سنوات",
    "القرار رقم ٤٧/١٢٣",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    result = replace_numbers_with_token(test)
    logger.info(f"   '{test}'")
    logger.info(f"   -> '{result}'")

# %% [markdown]
# ### 4.4 معالجة الكلمات النادرة

# %%
# ============================================================================
# اختياري 4.4: معالجة الكلمات النادرة
# ============================================================================

def build_vocabulary(dataset_dir: str, sample_size: int = 1000000) -> Counter:
    """
    بناء قاموس بتكرار الكلمات.
    
    المعاملات:
    -----------
    dataset_dir : str
        مسار البيانات
    sample_size : int
        عدد الأسطر للمعالجة
        
    ترجع:
    --------
    Counter
        عداد تكرار الكلمات
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
    استبدال الكلمات النادرة (أقل من حد التكرار) برمز خاص.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    vocab : Counter
        القاموس مع التكرارات
    threshold : int
        الحد الأدنى للتكرار لإبقاء الكلمة
    token : str
        رمز الاستبدال
        
    ترجع:
    --------
    str
        النص مع استبدال الكلمات النادرة
    """
    arabic_word_pattern = re.compile(r'[\u0600-\u06FF]+')
    
    def replace_if_rare(match):
        word = match.group(0)
        if vocab.get(word, 0) < threshold:
            return token
        return word
    
    return arabic_word_pattern.sub(replace_if_rare, text)


# ملاحظة: بناء القاموس مكلف حسابياً، سنوضح بمحاكاة
logger.section("🔧 اختياري: معالجة الكلمات النادرة")

logger.info("بناء القاموس عملية مكلفة حسابياً.")
logger.info("في الإنتاج، يجب بناء القاموس مرة واحدة وحفظه.")
logger.info("\nمثال الاستخدام:")
logger.info("   vocab = build_vocabulary(dataset_dir, sample_size=1000000)")
logger.info("   text = replace_rare_words(text, vocab, threshold=5)")

# %% [markdown]
# ### 4.5 تسوية طول الجمل

# %%
# ============================================================================
# اختياري 4.5: تسوية طول الجمل
# ============================================================================

def pad_sentence(text: str, target_length: int, pad_token: str = '<PAD>') -> str:
    """
    حشو الجملة لتصل للطول المستهدف.
    
    ملاحظة: هذا يتم عادة أثناء الـ batching، ليس المعالجة المسبقة.
    مدرج هنا للاكتمال.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    target_length : int
        عدد الكلمات المستهدف
    pad_token : str
        رمز الحشو
        
    ترجع:
    --------
    str
        النص المحشو
    """
    words = text.split()
    
    if len(words) >= target_length:
        return text
    
    padding = [pad_token] * (target_length - len(words))
    return text + ' ' + ' '.join(padding)


def split_into_chunks(text: str, chunk_size: int = 50, overlap: int = 10) -> List[str]:
    """
    تقسيم النص الطويل إلى أجزاء متداخلة.
    
    مفيد للمستندات الطويلة جداً حيث يجب الحفاظ على السياق.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    chunk_size : int
        الكلمات المستهدفة لكل جزء
    overlap : int
        عدد كلمات التداخل بين الأجزاء
        
    ترجع:
    --------
    List[str]
        قائمة أجزاء النص
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
        
        # تحريك البداية مع التداخل
        start = end - overlap
        if start >= len(words) - overlap:
            break
    
    return chunks


# اختبار الدوال
logger.section("🔧 اختياري: تسوية طول الجمل")

test_text = "هذا نص طويل يحتوي على العديد من الكلمات التي تحتاج إلى معالجة"
logger.info(f"الأصل: {len(test_text.split())} كلمة")

chunks = split_into_chunks(test_text, chunk_size=5, overlap=2)
logger.info(f"تم التقسيم إلى {len(chunks)} أجزاء:")
for i, chunk in enumerate(chunks):
    logger.info(f"   الجزء {i+1}: '{chunk}'")

# %% [markdown]
# ### 4.6 إزالة/استبدال المصطلحات الأجنبية

# %%
# ============================================================================
# اختياري 4.6: التعامل مع المصطلحات الأجنبية
# ============================================================================

def remove_document_references(text: str) -> str:
    """
    إزالة مراجع مستندات الأمم المتحدة (مثل A/47/10, S/RES/1234).
    
    المعاملات:
    -----------
    text : str
        النص المدخل
        
    ترجع:
    --------
    str
        النص بدون مراجع المستندات
    """
    # نمط مراجع المستندات
    doc_pattern = re.compile(r'[A-Z]/[A-Z0-9]+(?:/[A-Z0-9]+)*')
    return doc_pattern.sub('', text)


def replace_foreign_with_token(text: str, token: str = '<FOREIGN>') -> str:
    """
    استبدال المصطلحات الأجنبية (غير العربية) برمز خاص.
    
    المعاملات:
    -----------
    text : str
        النص المدخل
    token : str
        رمز الاستبدال
        
    ترجع:
    --------
    str
        النص مع استبدال المصطلحات الأجنبية
    """
    # نمط للكلمات اللاتينية (3+ حروف)
    foreign_pattern = re.compile(r'\b[A-Za-z]{3,}\b')
    return foreign_pattern.sub(token, text)


# اختبار الدوال
logger.section("🔧 اختياري: التعامل مع المصطلحات الأجنبية")

test_cases = [
    "الوثيقة A/47/10 المؤرخة",
    "برنامج UNDP للتنمية",
    "قرار مجلس الأمن S/RES/1234",
]

logger.info("حالات الاختبار:")
for test in test_cases:
    no_refs = remove_document_references(test)
    replaced = replace_foreign_with_token(test)
    logger.info(f"   الأصل:      '{test}'")
    logger.info(f"   إزالة مراجع:'{no_refs}'")
    logger.info(f"   استبدال:    '{replaced}'")

# %% [markdown]
# ---
# ## 5. الجزء 4: خط أنابيب المعالجة الكامل

# %%
# ============================================================================
# القسم 5: خط أنابيب المعالجة الكامل
# ============================================================================

@dataclass
class PreprocessingStats:
    """إحصائيات تم جمعها أثناء المعالجة."""
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
    خط أنابيب معالجة كامل لمجموعة بيانات الترقيم العربي.
    
    يوفر هذا الكلاس خط أنابيب قابل للتهيئة يمكن استخدامه مع
    كل من خطوات المعالجة الإلزامية والاختيارية.
    
    السمات:
    -----------
    config : PreprocessingConfig
        كائن الإعدادات المحتوي على جميع الخيارات
    stats : PreprocessingStats
        الإحصائيات المجمعة أثناء المعالجة
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        تهيئة المعالج بالإعدادات.
        
        المعاملات:
        -----------
        config : PreprocessingConfig
            كائن الإعدادات
        """
        self.config = config
        self.stats = PreprocessingStats()
        self.vocab = None  # لمعالجة الكلمات النادرة
        
        # تجميع الأنماط للكفاءة
        self._compile_patterns()
    
    def _compile_patterns(self):
        """تجميع أنماط Regex للكفاءة."""
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652]')
        self.alef_pattern = re.compile(r'[أإآٱ]')
        self.latin_pattern = re.compile(r'[A-Za-z]+')
        self.number_western = re.compile(r'[0-9]')
        self.consecutive_punct = re.compile(r'[،؛؟.,:;?!]{2,}')
        self.multi_space = re.compile(r' +')
        self.arabic_word = re.compile(r'[\u0600-\u06FF]+')
    
    def preprocess_line(self, text: str, apply_optional: bool = False) -> Optional[str]:
        """
        تطبيق خط أنابيب المعالجة الكامل على سطر واحد.
        
        المعاملات:
        -----------
        text : str
            سطر النص المدخل
        apply_optional : bool
            تطبيق الخطوات الاختيارية أم لا
            
        ترجع:
        --------
        Optional[str]
            النص المعالج، أو None إذا تم تصفية السطر
        """
        original = text
        
        # تتبع التغييرات للإحصائيات
        had_diacritics = bool(self.diacritics_pattern.search(text))
        had_alef_var = bool(self.alef_pattern.search(text))
        had_latin = bool(self.latin_pattern.search(text))
        had_consec_punct = bool(self.consecutive_punct.search(text))
        
        # ============================================
        # المعالجة الإلزامية
        # ============================================
        
        # 1. إزالة التشكيل
        if self.config.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)
            if had_diacritics:
                self.stats.diacritics_removed += 1
        
        # 2. توحيد الألف
        if self.config.normalize_alef:
            text = self.alef_pattern.sub('ا', text)
            if had_alef_var:
                self.stats.alef_normalized += 1
        
        # 3. توحيد الألف المقصورة (ى -> ي)
        if self.config.normalize_alef_maksura:
            text = text.replace('ى', 'ي')
        
        # 4. إزالة التطويل
        if self.config.remove_tatweel:
            text = text.replace('\u0640', '')
        
        # 5. توحيد الترقيم للعربي
        if self.config.unify_punctuation_to_arabic:
            for latin, arabic in LATIN_TO_ARABIC_PUNCT.items():
                if latin in text:
                    text = text.replace(latin, arabic)
                    self.stats.punct_normalized += 1
        
        # 6. توحيد الأرقام للعربية
        if self.config.unify_numbers_to_arabic:
            if self.number_western.search(text):
                text = text.translate(WESTERN_TO_ARABIC_NUMS)
                self.stats.numbers_normalized += 1
        
        # 7. إزالة الحروف اللاتينية
        if self.config.remove_latin_letters:
            if had_latin:
                text = self.latin_pattern.sub('', text)
                self.stats.latin_removed += 1
        
        # 8. إزالة الحروف خارج النطاق
        if self.config.remove_oov_chars:
            original_len = len(text)
            text = remove_oov_characters(text)
            if len(text) < original_len:
                self.stats.oov_removed += 1
        
        # 9. معالجة الترقيم المتتابع
        if self.config.handle_consecutive_punct:
            if had_consec_punct:
                text = handle_consecutive_punctuation_smart(text)
                self.stats.consecutive_punct_fixed += 1
        
        # 10. توحيد المسافات
        if self.config.normalize_whitespace:
            text = self.multi_space.sub(' ', text)
            text = text.strip()
            self.stats.whitespace_fixed += 1
        
        # 11. إضافة مسافات الترقيم
        if self.config.add_punct_spacing:
            text = add_punctuation_spacing(text)
        
        # ============================================
        # المعالجة الاختيارية
        # ============================================
        
        if apply_optional:
            # فصل واو العطف
            if self.config.separate_waw_conjunction:
                text = separate_waw_conjunction(text)
            
            # استبدال الأرقام برمز
            if self.config.replace_numbers_with_token:
                text = replace_numbers_with_token(text, '<NUM>')
            
            # إزالة المصطلحات الأجنبية
            if self.config.remove_foreign_terms:
                text = remove_document_references(text)
        
        # ============================================
        # التصفية
        # ============================================
        
        # تنظيف مسافات نهائي
        text = self.multi_space.sub(' ', text).strip()
        
        # التحقق من الفراغ
        if self.config.remove_empty_lines and not text:
            self.stats.empty_lines_removed += 1
            return None
        
        # التحقق من عدد الكلمات
        word_count = len(self.arabic_word.findall(text))
        
        # تصفية الجمل القصيرة
        if word_count < self.config.min_words:
            self.stats.short_lines_removed += 1
            return None
        
        # معالجة الجمل الطويلة
        if word_count > self.config.max_words:
            text = truncate_sentence(text, self.config.max_words)
            self.stats.long_lines_truncated += 1
        
        return text
    
    def process_dataset(self, input_dir: str, output_file: str, 
                        apply_optional: bool = False,
                        sample_size: Optional[int] = None) -> PreprocessingStats:
        """
        معالجة مجموعة البيانات بالكامل وحفظها في ملف.
        
        المعاملات:
        -----------
        input_dir : str
            مسار مجلد المدخلات
        output_file : str
            مسار ملف المخرجات
        apply_optional : bool
            تطبيق المعالجة الاختيارية أم لا
        sample_size : Optional[int]
            حد المعالجة لعدد معين من الأسطر (None = الكل)
            
        ترجع:
        --------
        PreprocessingStats
            إحصائيات المعالجة
        """
        logger.section("🚀 معالجة مجموعة البيانات")
        logger.info(f"المدخلات: {input_dir}")
        logger.info(f"المخرجات: {output_file}")
        logger.info(f"تطبيق الاختياري: {apply_optional}")
        
        # إعادة تعيين الإحصائيات
        self.stats = PreprocessingStats()
        
        # عد إجمالي الأسطر لشريط التقدم
        if sample_size is None:
            total_lines = count_total_lines(input_dir)
            logger.info(f"إجمالي الأسطر للمعالجة: {total_lines:,}")
        else:
            total_lines = sample_size
            logger.info(f"معالجة عينة من {total_lines:,} سطر")
        
        # إنشاء مجلد المخرجات إذا لزم الأمر
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # المعالجة والكتابة
        iterator = iter_dataset_lines(input_dir)
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=total_lines, desc="جاري المعالجة")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, line in enumerate(iterator):
                if sample_size and i >= sample_size:
                    break
                
                self.stats.total_input_lines += 1
                
                # معالجة السطر
                processed = self.preprocess_line(line, apply_optional)
                
                # الكتابة إذا لم يتم تصفيته
                if processed:
                    f.write(processed + '\n')
                    self.stats.total_output_lines += 1
        
        # تسجيل الإحصائيات
        self._log_statistics()
        
        return self.stats
    
    def _log_statistics(self):
        """تسجيل إحصائيات المعالجة."""
        logger.section("📊 إحصائيات المعالجة")
        
        logger.info(f"أسطر المدخلات:  {self.stats.total_input_lines:,}")
        logger.info(f"أسطر المخرجات:  {self.stats.total_output_lines:,}")
        
        kept_pct = self.stats.total_output_lines / max(self.stats.total_input_lines, 1) * 100
        logger.info(f"نسبة الإبقاء:   {kept_pct:.2f}%")
        
        logger.subsection("إحصائيات التصفية")
        logger.info(f"أسطر فارغة محذوفة:     {self.stats.empty_lines_removed:,}")
        logger.info(f"أسطر قصيرة محذوفة:     {self.stats.short_lines_removed:,}")
        logger.info(f"أسطر طويلة مقصوصة:     {self.stats.long_lines_truncated:,}")
        
        logger.subsection("إحصائيات التطبيع")
        logger.info(f"تشكيل محذوف:           {self.stats.diacritics_removed:,}")
        logger.info(f"ألف موحدة:             {self.stats.alef_normalized:,}")
        logger.info(f"ترقيم موحد:            {self.stats.punct_normalized:,}")
        logger.info(f"أرقام موحدة:           {self.stats.numbers_normalized:,}")
        logger.info(f"لاتيني محذوف:          {self.stats.latin_removed:,}")
        logger.info(f"OOV محذوف:             {self.stats.oov_removed:,}")
        logger.info(f"ترقيم متتابع مصلح:     {self.stats.consecutive_punct_fixed:,}")


# إنشاء المعالج
preprocessor = ArabicTextPreprocessor(config)
logger.success("تم تهيئة المعالج!")

# %% [markdown]
# ### اختبار خط الأنابيب الكامل

# %%
# ============================================================================
# اختبار خط الأنابيب الكامل
# ============================================================================

logger.section("🧪 اختبار خط الأنابيب الكامل")

# حالات اختبار تغطي مشكلات متنوعة
test_cases = [
    # تشكيل
    "الأُمَمُ المُتَّحِدَة مُنَظَّمَة دَوْلِيَّة.",
    
    # ترقيم مختلط
    "أولاً, ثانياً; ثالثاً?",
    
    # أشكال الألف
    "أحمد وإبراهيم وآدم",
    
    # أرقام
    "في عام 2024 وصل العدد إلى 100",
    
    # نص لاتيني
    "الوثيقة A/47/10 والقرار UNDP/2024",
    
    # ترقيم متتابع
    "ماذا؟؟؟ هذا صحيح...",
    
    # مشكلات مسافات
    "النص   مع    مسافات   كثيرة",
    
    # جملة قصيرة (يجب أن تحذف)
    "نعم",
    
    # مشكلات مجمعة
    "وَقَالَ أحمد: هذا مُهِمٌّ جِدَّاً,, والله!!",
]

logger.info("معالجة حالات الاختبار:\n")

for i, test in enumerate(test_cases, 1):
    result = preprocessor.preprocess_line(test, apply_optional=False)
    logger.info(f"الاختبار {i}:")
    logger.info(f"   الدخل:  '{test}'")
    if result:
        logger.info(f"   الخرج:  '{result}'")
    else:
        logger.info(f"   الخرج:  [تمت التصفية]")
    logger.info("")

# %% [markdown]
# ---
# ## 6. الجزء 5: الفحص بعد المعالجة

# %%
# ============================================================================
# القسم 6: الفحص بعد المعالجة
# ============================================================================

def inspect_preprocessed_data(file_path: str, sample_size: int = 100000) -> Dict:
    """
    فحص البيانات المعالجة للتحقق من الجودة.
    
    المعاملات:
    -----------
    file_path : str
        مسار الملف المعالج
    sample_size : int
        عدد الأسطر للفحص
        
    ترجع:
    --------
    Dict
        نتائج الفحص
    """
    logger.section("🔍 فحص ما بعد المعالجة")
    logger.info(f"جاري فحص: {file_path}")
    
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
        logger.error(f"الملف غير موجود: {file_path}")
        return stats
    
    # أنماط للفحص
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
            
            # تخزين عينة أسطر
            if len(stats['sample_lines']) < 10:
                stats['sample_lines'].append(line)
            
            # التحقق من المشكلات المتبقية
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
            
            # عد الترقيم
            for char in line:
                if char in VALID_PUNCTUATION:
                    stats['punctuation_dist'][char] += 1
    
    # عرض النتائج
    logger.subsection("إحصائيات أساسية")
    logger.info(f"إجمالي الأسطر: {stats['total_lines']:,}")
    logger.info(f"إجمالي الكلمات: {stats['total_words']:,}")
    logger.info(f"إجمالي الحروف: {stats['total_chars']:,}")
    
    if stats['word_counts']:
        word_arr = np.array(stats['word_counts'])
        logger.info(f"متوسط الكلمات/سطر: {np.mean(word_arr):.2f}")
        logger.info(f"الحد الأدنى: {np.min(word_arr)}")
        logger.info(f"الحد الأقصى: {np.max(word_arr)}")
    
    logger.subsection("فحص المشكلات المتبقية")
    total_issues = sum(stats['remaining_issues'].values())
    if total_issues == 0:
        logger.success("لم يتم العثور على مشكلات! البيانات نظيفة.")
    else:
        logger.warn(f"تم العثور على {total_issues} مشكلة محتملة:")
        for issue, count in stats['remaining_issues'].items():
            if count > 0:
                logger.info(f"   {issue}: {count:,} سطر")
    
    logger.subsection("توزيع الترقيم")
    for char, count in stats['punctuation_dist'].most_common():
        name = ARABIC_PUNCTUATION.get(char, 'غير معروف')
        logger.info(f"   '{char}' ({name}): {count:,}")
    
    logger.subsection("عينة أسطر")
    for i, line in enumerate(stats['sample_lines'][:5], 1):
        display = line[:80] + "..." if len(line) > 80 else line
        logger.info(f"   {i}. {display}")
    
    return stats


# سيتم تشغيل هذا بعد معالجة البيانات

# %% [markdown]
# ---
# ## 7. الجزء 6: حفظ البيانات المعالجة

# %%
# ============================================================================
# القسم 7: معالجة وحفظ البيانات
# ============================================================================

def run_preprocessing_pipeline(
    input_dir: str,
    output_dir: str,
    config: PreprocessingConfig,
    create_variants: bool = True,
    sample_size: Optional[int] = None
):
    """
    تشغيل خط أنابيب المعالجة الكامل وحفظ النتائج.
    
    ينشئ عدة متغيرات من المخرجات:
    1. mandatory_only.txt - معالجة إلزامية فقط
    2. with_waw_separation.txt - + فصل الواو
    3. with_number_tokens.txt - + استبدال الأرقام
    4. full_preprocessing.txt - جميع الخطوات
    
    المعاملات:
    -----------
    input_dir : str
        مسار مجلد المدخلات
    output_dir : str
        مسار مجلد المخرجات
    config : PreprocessingConfig
        كائن الإعدادات
    create_variants : bool
        إنشاء متغيرات متعددة أم لا
    sample_size : Optional[int]
        حد المعالجة (None = كامل)
    """
    logger.section("🚀 تشغيل خط أنابيب المعالجة")
    
    # إنشاء مجلد المخرجات
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================
    # المتغير 1: المعالجة الإلزامية فقط
    # ============================================
    logger.subsection("المتغير 1: المعالجة الإلزامية فقط")
    
    output_file_mandatory = os.path.join(output_dir, "mandatory_only.txt")
    
    # ضمان إيقاف الخطوات الاختيارية
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
    
    # حفظ الإحصائيات
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
    # المتغير 2: مع فصل الواو
    # ============================================
    logger.subsection("المتغير 2: مع فصل الواو")
    
    config.separate_waw_conjunction = True
    preprocessor = ArabicTextPreprocessor(config)
    
    output_file_waw = os.path.join(output_dir, "with_waw_separation.txt")
    stats_waw = preprocessor.process_dataset(
        input_dir,
        output_file_waw,
        apply_optional=True,
        sample_size=sample_size
    )
    
    config.separate_waw_conjunction = False  # إعادة ضبط
    
    # ============================================
    # المتغير 3: مع رموز الأرقام
    # ============================================
    logger.subsection("المتغير 3: مع رموز الأرقام")
    
    config.replace_numbers_with_token = True
    preprocessor = ArabicTextPreprocessor(config)
    
    output_file_nums = os.path.join(output_dir, "with_number_tokens.txt")
    stats_nums = preprocessor.process_dataset(
        input_dir,
        output_file_nums,
        apply_optional=True,
        sample_size=sample_size
    )
    
    config.replace_numbers_with_token = False  # إعادة ضبط
    
    # ============================================
    # المتغير 4: المعالجة الكاملة
    # ============================================
    logger.subsection("المتغير 4: المعالجة الكاملة (كل الخيارات)")
    
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
    
    logger.section("✅ تمت المعالجة")
    logger.info(f"تم حفظ الملفات في: {output_dir}")
    logger.info("الملفات التي تم إنشاؤها:")
    logger.info(f"   1. mandatory_only.txt")
    logger.info(f"   2. with_waw_separation.txt")
    logger.info(f"   3. with_number_tokens.txt")
    logger.info(f"   4. full_preprocessing.txt")


# %%
# ============================================================================
# تنفيذ المعالجة
# ============================================================================

# إعدادات التنفيذ
EXECUTE_FULL_PIPELINE = True  # اجعلها True لتنفيذ المعالجة
SAMPLE_SIZE_FOR_TEST = 100000  # اجعلها None للمعالجة الكاملة

if EXECUTE_FULL_PIPELINE:
    logger.section("⚡ تنفيذ خط أنابيب المعالجة")
    
    # للاختبار، استخدم عينة
    sample = SAMPLE_SIZE_FOR_TEST  # غيرها لـ None للمعالجة الكاملة
    
    run_preprocessing_pipeline(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        config=config,
        create_variants=True,
        sample_size=sample
    )
    
    # فحص الناتج الإلزامي
    mandatory_file = os.path.join(config.output_dir, "mandatory_only.txt")
    if os.path.exists(mandatory_file):
        post_stats = inspect_preprocessed_data(mandatory_file, sample_size=50000)
else:
    logger.info("تم تخطي التنفيذ. اجعل EXECUTE_FULL_PIPELINE = True للتنفيذ.")

# %% [markdown]
# ---
# ## 8. الملخص والتوصيات

# %%
# ============================================================================
# القسم 8: الملخص والتوصيات
# ============================================================================

logger.section("📋 ملخص المعالجة والتوصيات")

summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ملخص خط أنابيب المعالجة                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  خطوات المعالجة الإلزامية (تطبق دائماً):                                     ║
║  ──────────────────────────────────────                                      ║
║  ✅ إزالة التشكيل (الحركات)                                                  ║
║  ✅ توحيد أشكال الألف (أإآٱ -> ا)                                            ║
║  ✅ توحيد الألف المقصورة (ى -> ي)                                            ║
║  ✅ إزالة التطويل (ـ)                                                        ║
║  ✅ توحيد الترقيم للعربي (,;? -> ،؛؟)                                        ║
║  ✅ توحيد الأرقام للعربية (0-9 -> ٠-٩)                                       ║
║  ✅ إزالة الحروف اللاتينية                                                   ║
║  ✅ إزالة الحروف خارج النطاق (OOV)                                           ║
║  ✅ معالجة الترقيم المتتابع                                                  ║
║  ✅ توحيد المسافات                                                           ║
║  ✅ إضافة مسافات الترقيم                                                     ║
║  ✅ تصفية الأسطر الفارغة                                                     ║
║  ✅ تصفية الجمل القصيرة (<3 كلمات)                                           ║
║  ✅ قص الجمل الطويلة (>100 كلمة)                                             ║
║                                                                              ║
║  خطوات المعالجة الاختيارية (للتجريب):                                        ║
║  ───────────────────────────────────                                         ║
║  ⚙️  فصل واو العطف (و + كلمة -> و كلمة)                                      ║
║  ⚙️  استبدال الأرقام برمز (<NUM>)                                            ║
║  ⚙️  إزالة مراجع المستندات (A/47/10)                                         ║
║  ⚙️  إزالة/تمييز كلمات التوقف                                                ║
║  ⚙️  استبدال الكلمات النادرة (<UNK>)                                         ║
║                                                                              ║
║  الملفات الناتجة:                                                            ║
║  ───────────────                                                             ║
║  📁 preprocessed_data/                                                       ║
║     ├── mandatory_only.txt      (موصى به لمعظم التجارب)                      ║
║     ├── with_waw_separation.txt (لاختبار تأثير فصل الواو)                    ║
║     ├── with_number_tokens.txt  (لاختبار تأثير استبدال الأرقام)              ║
║     └── full_preprocessing.txt  (كل الخطوات مطبقة)                           ║
║                                                                              ║
║  التوصيات:                                                                   ║
║  ─────────                                                                   ║
║  1. ابدأ بـ mandatory_only.txt للتجارب الأساسية                              ║
║  2. استخدم with_waw_separation.txt إذا ساعد فصل الواو                        ║
║  3. جرب with_number_tokens.txt إذا سببت الأرقام مشكلات                       ║
║  4. قارن النتائج لتحديد المعالجة الأمثل                                      ║
║                                                                              ║
║  الخطوات التالية:                                                            ║
║  ───────────────                                                             ║
║  1. ترميز البيانات المعالجة (Tokenization)                                   ║
║  2. إنشاء تقسيمات التدريب/التحقق/الاختبار                                    ║
║  3. توليد التصنيفات (Labels) لمهمة التسلسل-إلى-تسلسل                         ║
║  4. تدريب وتقييم النماذج                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print(summary)

# %%
# ============================================================================
# أداة توليد التصنيفات (Label Generation)
# ============================================================================

def generate_labels_for_line(text: str) -> Tuple[List[str], List[int]]:
    """
    توليد تسلسلات الكلمات والتصنيفات لسطر معالج.
    
    هذا هو التنسيق المطلوب لتدريب التسلسل-إلى-تسلسل.
    
    المعاملات:
    -----------
    text : str
        سطر نص معالج مسبقاً
        
    ترجع:
    --------
    Tuple[List[str], List[int]]
        (كلمات، تصنيفات) حيث تشير التصنيفات للترقيم بعد كل كلمة
        
    خريطة التصنيفات:
    - 0: لا ترقيم (O)
    - 1: نقطة (.)
    - 2: فاصلة عربية (،)
    - 3: علامة استفهام (؟)
    - 4: فاصلة منقوطة (؛)
    - 5: نقطتان رأسيتان (:)
    - 6: علامة تعجب (!)
    """
    LABEL_MAP = {
        'O': 0,    # لا ترقيم
        '.': 1,    # نقطة
        '،': 2,    # فاصلة عربية
        '؟': 3,    # علامة استفهام
        '؛': 4,    # فاصلة منقوطة
        ':': 5,    # نقطتان رأسيتان
        '!': 6,    # علامة تعجب
    }
    
    words = []
    labels = []
    
    # نمط للكلمات العربية
    word_pattern = re.compile(r'[\u0600-\u06FF٠-٩]+')
    
    # العثور على الكلمات ومواقعها
    for match in word_pattern.finditer(text):
        word = match.group()
        end_pos = match.end()
        
        # البحث عن ترقيم مباشرة بعد الكلمة
        remaining = text[end_pos:].lstrip()
        
        if remaining and remaining[0] in LABEL_MAP:
            label = LABEL_MAP[remaining[0]]
        else:
            label = LABEL_MAP['O']
        
        words.append(word)
        labels.append(label)
    
    return words, labels


# اختبار توليد التصنيفات
logger.section("🏷️ مثال توليد التصنيفات")

test_lines = [
    "هذا نص عربي، يحتوي على علامات ترقيم.",
    "ما هو السؤال؟",
    "أولاً؛ ثانياً؛ ثالثاً.",
]

logger.info("خريطة التصنيفات:")
logger.info("   0: لا ترقيم (O)")
logger.info("   1: نقطة (.)")
logger.info("   2: فاصلة (،)")
logger.info("   3: استفهام (؟)")
logger.info("   4: فاصلة منقوطة (؛)")
logger.info("   5: نقطتان (:)")
logger.info("   6: تعجب (!)")
logger.info("")

for text in test_lines:
    words, labels = generate_labels_for_line(text)
    logger.info(f"النص: {text}")
    logger.info(f"الكلمات: {words}")
    logger.info(f"التصنيفات: {labels}")
    logger.info("")

# %%
logger.section("✅ اكتمل دفتر المعالجة المسبقة")
logger.info("تم تعريف جميع دوال وخطوات المعالجة.")
logger.info("شغل الخط الكامل بوضع EXECUTE_FULL_PIPELINE = True لمعالجة كامل البيانات.")