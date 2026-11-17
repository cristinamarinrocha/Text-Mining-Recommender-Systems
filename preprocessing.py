"""
preprocessing.py
================
STEP 0 - Loading the data
STEP 1 - Generic normalisation
STEP 2 - Tokenisation

Everything in this file is "product-agnostic": the rules only deal with text
formatting and units, never with TV-specific attributes.
"""

import html
import json
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------
# STEP 0 - Load the data
# --------------------------------------------------------------------------
@dataclass
class Product:
    """One product description from one Web shop."""
    shop: str
    title: str
    features: dict      # "featuresMap" in the JSON data
    model_id: str       # only used for evaluation


def load_products(path):
    """
    The JSON file is a dictionary with structure:  {modelID: [product, product, ...]}
    - Products with same modelID are duplicates of each other.
    Flatten it into one list of Product objects.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    products = [] # initialize
    # raw.items() gives pairs (model_id, descriptions)
    # descriptions is the list of all product descriptions with that model ID, and d is one of them
    for model_id, descriptions in raw.items():
        for d in descriptions: # turn every d into a product and out it on flat list
            products.append(Product(shop=d["shop"],
                                    title=d["title"],
                                    features=d["featuresMap"],
                                    model_id=model_id))
    return products

# -------------------------------------------------------------------------
# STEP 1 - Generic normalisation applied to every title and value
# -------------------------------------------------------------------------
# use re rules ot normalize text of names and features (order matters)
# - convert to lower case and decode HTML symbols
# - remove shop names
# - unify unit spellings
# - remove punctuation inside tokens

SHOP_NAMES = re.compile(r"newegg\.com|newegg|best\s?buy|thenerds\.net|thenerds|amazon\.com|amazon")
INCH = re.compile(r"(\d)\s*-?\s*(?:\"|''|”|inches|inch|in\.)")
HERTZ = re.compile(r"(\d)\s*-?\s*(?:hertz|hz)\b")
POUND = re.compile(r"(\d)\s*-?\s*(?:pounds|pound|lbs|lb)\b\.?")

# A - / or . with a letter/digit on BOTH sides, e.g. the "-" in "lc-90le657u".
# (?<=...) and (?=...) are "look-arounds": they check what is just before and
# after the symbol without removing those characters themselves.
INNER_PUNCTUATION = re.compile(r"(?<=[a-z0-9])[-/.](?=[a-z0-9])")


def normalise(text):
    """Apply the generic normalisation rules of Step 1 to one string."""
    text = html.unescape(text)              # "&quot;" -> '"' , "&amp;" -> "&"
    text = text.lower()                     # case
    text = SHOP_NAMES.sub(" ", text)        # shop names
    text = INCH.sub(r"\1inch ", text)       # units
    text = HERTZ.sub(r"\1hz ", text)
    text = POUND.sub(r"\1lb ", text)
    text = INNER_PUNCTUATION.sub("", text)  # "lc-90l357u" -> "lc90l357u"
    return text


# -------------------------------------------------------------------------
# STEP 2 - Tokenisation  tau(.)
# -------------------------------------------------------------------------
# After Step 1 a token is just a run of letters and digits.
TOKEN = re.compile(r"[a-z0-9]+")

def tokenise(text):
    """Split a normalised string into its set of tokens."""
    return set(TOKEN.findall(text))

def product_tokens(product):
    """
    T_i in the paper: tokens of the title  UNION  tokens of every attribute VALUE.
    """
    tokens = tokenise(normalise(product.title))
    for value in product.features.values():
        tokens |= tokenise(normalise(value)) # |= updates the union
    return tokens

