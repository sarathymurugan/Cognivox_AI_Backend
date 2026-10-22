import re

ENGLISH_FILLERS = {"uh", "um", "er", "ah", "hmm", "uhh"}
ENGLISH_PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "something", "thing", "stuff", "wherever"
}

TAMIL_FILLERS = {"வந்து", "அதாவது", "ம்ம்", "ஆமா", "என்னன்னா", "சரி", "அப்புறம்"}
TAMIL_PRONOUNS = {
    "நான்", "என்", "எனக்கு", "நாம்", "நாங்கள்", "நீ", "நீங்கள்",
    "அவன்", "அவள்", "அவர்", "அவர்கள்", "அது", "இது", "அவை", "இவை"
}


def extract_linguistic_biomarkers(transcript: str, language: str = "en") -> dict:
    """
    Extracts linguistic MCI biomarkers from English, Tamil, or code-switched transcripts:
    - Filler word rate (hesitation / disfluencies)
    - Pronoun usage percentage (vague reference substitution)
    - Lexical diversity (Type-Token Ratio)
    """

    attention_branch = "dot_product_tamil" if language == "ta" else "additive_english"

    
    if not transcript or not transcript.strip():
        return {
            "language": language,
            "total_words": 0,
            "unique_words": 0,
            "lexical_diversity_ttr": 0.0,
            "filler_count": 0,
            "filler_percentage": 0.0,
            "pronoun_count": 0,
            "pronoun_percentage": 0.0,
            "attention_branch": attention_branch
        }

    # Match both English words and Tamil Unicode characters (\u0B80-\u0BFF)
    words = re.findall(r"[\w\u0B80-\u0BFF]+", transcript.lower())
    total_words = len(words)

    if total_words == 0:
        return {
            "language": language,
            "total_words": 0,
            "unique_words": 0,
            "lexical_diversity_ttr": 0.0,
            "filler_count": 0,
            "filler_percentage": 0.0,
            "repetition_count": 0,
            "pronoun_count": 0,
            "pronoun_percentage": 0.0,
            "attention_branch": attention_branch
        }

    unique_words = set(words)
    ttr = len(unique_words) / total_words

    all_fillers = ENGLISH_FILLERS | TAMIL_FILLERS
    all_pronouns = ENGLISH_PRONOUNS | TAMIL_PRONOUNS

    filler_count = sum(1 for w in words if w in all_fillers)
    pronoun_count = sum(1 for w in words if w in all_pronouns)

    # Detect immediate 1-word and 2-word stutters/repetitions (e.g., "the the" or "in the in the")
    repetition_count = 0
    for i in range(1, total_words):
        if words[i] == words[i - 1]:
            repetition_count += 1
        elif i >= 3 and words[i - 1:i + 1] == words[i - 3:i - 1]:
            repetition_count += 1

    # Total disfluencies = vocalized fillers + stutters/repetitions
    total_disfluencies = filler_count + repetition_count

    return {
        "language": language,
        "total_words": int(total_words),
        "unique_words": int(len(unique_words)),
        "lexical_diversity_ttr": round(float(ttr), 4),
        "filler_count": int(filler_count),
        "repetition_count": int(repetition_count),
        "filler_percentage": round(float(filler_count / total_words * 100), 2),
        "pronoun_count": int(pronoun_count),
        "pronoun_percentage": round(float(pronoun_count / total_words * 100), 2),
        "attention_branch": attention_branch
    }