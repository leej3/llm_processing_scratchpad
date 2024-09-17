from pydantic import BaseModel, Field
import re
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path

def make_uid_path_safe(uid: str) -> str:
    """
    Sanitizes a given string to make it safe for use as a file path.

    Args:
    - uid (str): The original string that needs to be sanitized.

    Returns:
    - str: A sanitized string safe for use as a file path.
    """
    # Define a mapping of unsafe characters to safe replacements
    unsafe_to_safe = {
        '/': '_slash_',
        '\\': '_backslash_',
        ':': '_colon_',
        '*': '_asterisk_',
        '?': '_question_',
        '"': '_quote_',
        '<': '_lt_',
        '>': '_gt_',
        '|': '_pipe_'
    }

    # Replace unsafe characters with their safe replacements
    safe_uid = uid
    for unsafe_char, safe_char in unsafe_to_safe.items():
        safe_uid = safe_uid.replace(unsafe_char, safe_char)

    # Remove leading/trailing whitespace
    safe_uid = safe_uid.strip()

    # Replace multiple consecutive spaces or underscores with a single underscore
    safe_uid = re.sub(r"[\s_]+", "_", safe_uid)

    # Return the sanitized UID
    return safe_uid

def revert_uid_path_safe(safe_uid: str) -> str:
    """
    Reverts a sanitized string back to its original form.

    Args:
    - safe_uid (str): The sanitized string that needs to be reverted.

    Returns:
    - str: The original string before sanitization.
    """
    # Define a mapping of safe replacements back to unsafe characters
    safe_to_unsafe = {
        '_slash_': '/',
        '_backslash_': '\\',
        '_colon_': ':',
        '_asterisk_': '*',
        '_question_': '?',
        '_quote_': '"',
        '_lt_': '<',
        '_gt_': '>',
        '_pipe_': '|'
    }

    # Revert safe replacements back to unsafe characters
    original_uid = safe_uid
    for safe_char, unsafe_char in safe_to_unsafe.items():
        original_uid = original_uid.replace(safe_char, unsafe_char)

    return original_uid

def extract_key_components(filename):
    # Split filename into components, remove common fillers, and clean up
    parts = filename.name.split('_-_')
    authors = parts[0].replace('_', ' ')
    title = parts[1] if len(parts) > 1 else ''
    return authors.lower(), title.lower()

def simplify_string(text):
    """Simplify the string by removing special characters and converting to lowercase."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower())

# Improved matching function using string similarity
def find_best_match(layer_title_authors,authors_key, title_key):
    best_match = None
    highest_score = 0

    for _, row in layer_title_authors.iterrows():
        # Compare simplified strings
        authors_similarity = SequenceMatcher(None, authors_key, row['simplified_authors']).ratio()
        title_similarity = SequenceMatcher(None, title_key, row['simplified_title']).ratio()

        # Calculate a combined score and update the best match if this one is higher
        combined_score = (authors_similarity + title_similarity) / 2
        if combined_score > highest_score and combined_score > 0.5:  # Set a threshold for relevance
            highest_score = combined_score
            best_match = row['Link']

    return best_match

def get_layer_paper_mapping(layer_files, layer_title_authors):

    # Simplify titles and authors for easier matching
    layer_title_authors['simplified_authors'] = layer_title_authors['Authors'].fillna('').apply(simplify_string)
    layer_title_authors['simplified_title'] = layer_title_authors['Title'].fillna('').apply(simplify_string)


    # Iterate through filenames and match with title_authors DataFrame
    filename_to_link = {}
    for filename in layer_files:
        authors_key, title_key = extract_key_components(filename)
        authors_key = simplify_string(authors_key)
        title_key = simplify_string(title_key)

        # Find the best match using the improved matching function
        link = find_best_match(layer_title_authors,authors_key, title_key)

        # Store the link if a match is found
        if link:
            filename_to_link[filename] = link
    return filename_to_link

def append_filenames_to_dataframe(df, filename_to_link_mapping):
    """
    Append filenames to the dataframe based on the filename to link mapping.
    Drop rows that do not have a matching link in the mapping.

    Args:
    df (pd.DataFrame): The dataframe containing titles and authors.
    filename_to_link_mapping (dict): A dictionary mapping filenames to their corresponding links.

    Returns:
    pd.DataFrame: The updated dataframe with an additional 'Filename' column.
    """
    # Create a reverse mapping from link to filename
    link_to_filename = {link: filename for filename, link in filename_to_link_mapping.items()}

    # Add a new column 'Filename' using the mapping
    df['Filename'] = df['Link'].map(link_to_filename)

    # Drop rows where 'Filename' is NaN (i.e., no match was found)
    df = df.dropna(subset=['Filename'])

    return df



def cleanup():
    """Moves pdfs from nimh and layer directories into a combined directory.
    Writes out a tsv with the appropriate harmonized metadata. Output data requires 
    - filename
    - manual data rating
    - year
    - journal
    - doi (though could be extracted from filename)
    - 'Data availability (3=available online, 2 available upon request, 1, no sharing or no statement, 0 not applicable)'
    - Comment
    """
    layer_files = list(Path("tempdata/layerfMRI_2024-08").glob("*.pdf"))
    layer_title_authors = pd.read_csv("tempdata/layer-fMRI papers in humans - Papers.tsv",delimiter="\t",skiprows=0)
    layer_df = append_filenames_to_dataframe(
        layer_title_authors,
        get_layer_paper_mapping(layer_files, layer_title_authors),
    )
    return layer_df
