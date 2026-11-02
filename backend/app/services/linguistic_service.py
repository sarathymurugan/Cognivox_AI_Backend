# import re

# ENGLISH_FILLERS = {"uh", "um", "er", "ah", "hmm", "uhh"}
# ENGLISH_PRONOUNS = {
#     "i", "me", "my", "mine", "we", "us", "our", "you", "your",
#     "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
#     "this", "that", "these", "those", "something", "thing", "stuff", "wherever"
# }

# TAMIL_FILLERS = {"வந்து", "அதாவது", "ம்ம்", "ஆமா", "என்னன்னா", "சரி", "அப்புறம்"}
# TAMIL_PRONOUNS = {
#     "நான்", "என்", "எனக்கு", "நாம்", "நாங்கள்", "நீ", "நீங்கள்",
#     "அவன்", "அவள்", "அவர்", "அவர்கள்", "அது", "இது", "அவை", "இவை"
# }


# def extract_linguistic_biomarkers(transcript: str, language: str = "en") -> dict:
#     """
#     Extracts linguistic MCI biomarkers from English, Tamil, or code-switched transcripts:
#     - Filler word rate (hesitation / disfluencies)
#     - Pronoun usage percentage (vague reference substitution)
#     - Lexical diversity (Type-Token Ratio)
#     """

#     attention_branch = "dot_product_tamil" if language == "ta" else "additive_english"

    
#     if not transcript or not transcript.strip():
#         return {
#             "language": language,
#             "total_words": 0,
#             "unique_words": 0,
#             "lexical_diversity_ttr": 0.0,
#             "filler_count": 0,
#             "filler_percentage": 0.0,
#             "pronoun_count": 0,
#             "pronoun_percentage": 0.0,
#             "attention_branch": attention_branch
#         }

#     # Match both English words and Tamil Unicode characters (\u0B80-\u0BFF)
#     words = re.findall(r"[\w\u0B80-\u0BFF]+", transcript.lower())
#     total_words = len(words)

#     if total_words == 0:
#         return {
#             "language": language,
#             "total_words": 0,
#             "unique_words": 0,
#             "lexical_diversity_ttr": 0.0,
#             "filler_count": 0,
#             "filler_percentage": 0.0,
#             "repetition_count": 0,
#             "pronoun_count": 0,
#             "pronoun_percentage": 0.0,
#             "attention_branch": attention_branch
#         }

#     unique_words = set(words)
#     ttr = len(unique_words) / total_words

#     all_fillers = ENGLISH_FILLERS | TAMIL_FILLERS
#     all_pronouns = ENGLISH_PRONOUNS | TAMIL_PRONOUNS

#     filler_count = sum(1 for w in words if w in all_fillers)
#     pronoun_count = sum(1 for w in words if w in all_pronouns)

#     # Detect immediate 1-word and 2-word stutters/repetitions (e.g., "the the" or "in the in the")
#     repetition_count = 0
#     for i in range(1, total_words):
#         if words[i] == words[i - 1]:
#             repetition_count += 1
#         elif i >= 3 and words[i - 1:i + 1] == words[i - 3:i - 1]:
#             repetition_count += 1

#     # Total disfluencies = vocalized fillers + stutters/repetitions
#     total_disfluencies = filler_count + repetition_count

#     return {
#         "language": language,
#         "total_words": int(total_words),
#         "unique_words": int(len(unique_words)),
#         "lexical_diversity_ttr": round(float(ttr), 4),
#         "filler_count": int(filler_count),
#         "repetition_count": int(repetition_count),
#         "filler_percentage": round(float(filler_count / total_words * 100), 2),
#         "pronoun_count": int(pronoun_count),
#         "pronoun_percentage": round(float(pronoun_count / total_words * 100), 2),
#         "attention_branch": attention_branch
#     }





# backend/app/services/linguistic_service.py
import re

# Pure vocalized fillers (Farzana et al., 2022 / DementiaBank FREQ)
ENGLISH_FILLERS = {"uh", "um", "er", "ah", "hmm", "uhh", "umm"}
ENGLISH_PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "something", "thing", "things", "stuff", "wherever"
}

TAMIL_FILLERS = {"வந்து", "அதாவது", "ம்ம்", "ஆமா", "என்னன்னா", "அப்புறம்"}
TAMIL_PRONOUNS = {
    "நான்", "என்", "எனக்கு", "நாம்", "நாங்கள்", "நீ", "நீங்கள்",
    "அவன்", "அவள்", "அவர்", "அவர்கள்", "அது", "இது", "அவை", "இவை", "ஏதோ"
}

# Bilingual Core Lexicon (CoreLex) lemma checklists (Dalton et al., 2020 / DementiaBank)
CORE_LEXICON_EN = {
    "boy", "girl", "man", "woman", "mother", "father", "child", "children", "dog", "cat",
    "car", "window", "tree", "water", "sink", "cookie", "jar", "stool", "plate", "drive",
    "look", "see", "stand", "fall", "climb", "overflow", "tea", "cup", "water", "milk",
    "sugar", "boil", "pour", "drink", "work", "home", "family", "morning", "day", "go"
}

CORE_LEXICON_TA = {
    "பையன்", "பொண்ணு", "சிறுவன்", "சிறுமி", "அம்மா", "அப்பா", "குழந்தை", "நாய்", "பூனை",
    "கார்", "ஜன்னல்", "மரம்", "தண்ணீர்", "தட்டு", "நாற்காலி", "பார்", "விழு", "ஏறு",
    "டீ", "தேநீர்", "பால்", "சர்க்கரை", "கொதிக்க", "வடிகட்டி", "குடி", "வேலை", "வீடு", "ஊர்", "குடும்பம்"
}

# Standard DementiaBank Examiner Prompts to strip if present in clinical audio files
EXAMINER_PROMPTS = [
    r"do you need to like zoom in or anything\??",
    r"okay,? so i'?m going to have you tell me a story with a beginning,? middle,? (in|and) an end\.?",
    r"please tell me everything you see going on in this picture\.?",
    r"tell me the story with a beginning,? a middle,? and an end\.?",
]


def clean_examiner_speech(transcript: str) -> str:
    """Removes standardized DementiaBank examiner instructions from the start of a transcript."""
    cleaned = transcript
    for pattern in EXAMINER_PROMPTS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_linguistic_biomarkers(
    transcript: str,
    language: str = "en",
    duration_seconds: float = 0.0
) -> dict:
    """
    Extracts CMRG + DementiaBank (EVAL-D & CoreLex) linguistic biomarkers:
    - Filler word rate & word repetitions (Farzana et al., 2022)
    - Pronoun usage percentage (Delaware Corpus MCI marker: 14.2% vs 11.9%)
    - Lexical diversity (Type-Token Ratio)
    - Mean Length of Utterance in words (MLU - Delaware Corpus MCI marker: 9.9 vs 11.9)
    - Core Lexicon (CoreLex) count & CoreLex/min efficiency (Dalton et al., 2020)
    """
    attention_branch = "dot_product_tamil" if language == "ta" else "additive_english"

    if not transcript or not transcript.strip():
        return {
            "language": language,
            "total_words": 0,
            "unique_words": 0,
            "lexical_diversity_ttr": 0.0,
            "mlu_words": 0.0,
            "words_per_minute": 0.0,
            "filler_count": 0,
            "repetition_count": 0,
            "filler_percentage": 0.0,
            "pronoun_count": 0,
            "pronoun_percentage": 0.0,
            "core_lexicon_count": 0,
            "core_lexicon_per_min": 0.0,
            "attention_branch": attention_branch
        }

    # Strip examiner instructions if present (e.g., TAUKADIAL / DementiaBank recordings)
    participant_text = clean_examiner_speech(transcript)
    if not participant_text:
        participant_text = transcript

    # Split into utterances (. ! ?) to calculate Mean Length of Utterance (MLU)
    utterances = [
        u.strip() for u in re.split(r"[.!?।]+", participant_text)
        if len(re.findall(r"[\w\u0B80-\u0BFF]+", u)) > 0
    ]
    num_utterances = max(1, len(utterances))

    words = re.findall(r"[\w\u0B80-\u0BFF]+", participant_text.lower())
    total_words = len(words)

    if total_words == 0:
        return {
            "language": language,
            "total_words": 0,
            "unique_words": 0,
            "lexical_diversity_ttr": 0.0,
            "mlu_words": 0.0,
            "words_per_minute": 0.0,
            "filler_count": 0,
            "repetition_count": 0,
            "filler_percentage": 0.0,
            "pronoun_count": 0,
            "pronoun_percentage": 0.0,
            "core_lexicon_count": 0,
            "core_lexicon_per_min": 0.0,
            "attention_branch": attention_branch
        }

    unique_words = set(words)
    ttr = len(unique_words) / total_words
    mlu_words = total_words / num_utterances

    all_fillers = ENGLISH_FILLERS | TAMIL_FILLERS
    all_pronouns = ENGLISH_PRONOUNS | TAMIL_PRONOUNS
    all_core_lex = CORE_LEXICON_EN | CORE_LEXICON_TA

    filler_count = sum(1 for w in words if w in all_fillers)
    pronoun_count = sum(1 for w in words if w in all_pronouns)

    # Core Lexicon (CoreLex): Unique normative checklist items produced by participant
    core_lex_matched = sum(1 for uw in unique_words if any(uw.startswith(cl) for cl in all_core_lex))

    # Immediate 1-word and 2-word stutters/repetitions
    repetition_count = 0
    for i in range(1, total_words):
        if words[i] == words[i - 1]:
            repetition_count += 1
        elif i >= 3 and words[i - 1:i + 1] == words[i - 3:i - 1]:
            repetition_count += 1

    total_disfluencies = filler_count + repetition_count

    # Derived Efficiency Measures (Words/min and CoreLex/min - Dalton et al., 2020)
    duration_min = (duration_seconds / 60.0) if duration_seconds and duration_seconds > 0 else (total_words / 110.0)
    words_per_min = total_words / duration_min if duration_min > 0 else 0.0
    core_lex_per_min = core_lex_matched / duration_min if duration_min > 0 else 0.0

    return {
        "language": language,
        "total_words": int(total_words),
        "unique_words": int(len(unique_words)),
        "utterance_count": int(num_utterances),
        "mlu_words": round(float(mlu_words), 2),
        "words_per_minute": round(float(words_per_min), 1),
        "lexical_diversity_ttr": round(float(ttr), 4),
        "filler_count": int(filler_count),
        "repetition_count": int(repetition_count),
        "filler_percentage": round(float(total_disfluencies / total_words * 100), 2),
        "pronoun_count": int(pronoun_count),
        "pronoun_percentage": round(float(pronoun_count / total_words * 100), 2),
        "core_lexicon_count": int(core_lex_matched),
        "core_lexicon_per_min": round(float(core_lex_per_min), 2),
        "attention_branch": attention_branch
    }