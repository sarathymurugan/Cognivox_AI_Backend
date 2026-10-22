# def calculate_mci_risk_score(
#     acoustic_features: dict,
#     linguistic_features: dict = None,
#     age: int = None,
#     gender: str = None
# ) -> dict:
#     """
#     Fuses acoustic biomarkers (pause rate, pitch variability, jitter, shimmer)
#     and linguistic biomarkers (filler rate, pronoun overuse, lexical diversity)
#     to generate a normalized MCI Risk Score (0-100) and PHC referral guidance.
#     """
#     risk_points = 0.0
#     max_points = 0.0
#     flags = []

#     # 1. Pause & Hesitation Risk (Max 30 points)
#     pause_info = acoustic_features.get("pause", {})
#     silence_pct = pause_info.get("silence_percentage", 0.0) or 0.0
#     longest_pause = pause_info.get("longest_pause_seconds", 0.0) or 0.0
#     max_points += 30.0

#     if silence_pct > 35.0 or longest_pause > 2.0:
#         risk_points += 28.0
#         flags.append("High speech hesitation and prolonged pauses detected (>35% silence).")
#     elif silence_pct > 20.0 or longest_pause > 1.2:
#         risk_points += 15.0
#         flags.append("Moderate pause frequency indicating potential word-finding delay.")
#     else:
#         risk_points += 4.0

#     # 2. Vocal Stability Risk: Jitter & Shimmer (Max 25 points)
#     js_info = acoustic_features.get("jitter_shimmer", {})
#     jitter = js_info.get("jitter")
#     shimmer = js_info.get("shimmer")
#     if jitter is not None and shimmer is not None:
#         max_points += 25.0
#         if jitter > 0.02 or shimmer > 0.08:
#             risk_points += 22.0
#             flags.append("Elevated vocal instability (jitter/shimmer above normal range).")
#         elif jitter > 0.012 or shimmer > 0.05:
#             risk_points += 12.0
#             flags.append("Mild vocal cord micro-tremor (possible chronic stress indicator).")
#         else:
#             risk_points += 3.0

#     # 3. Prosody / Pitch Flattening Risk (Max 20 points)
#     pitch_info = acoustic_features.get("pitch", {})
#     pitch_std = pitch_info.get("std_hz")
#     if pitch_std is not None:
#         max_points += 20.0
#         if pitch_std < 12.0:
#             risk_points += 18.0
#             flags.append("Flat prosody / monotone vocal delivery detected.")
#         elif pitch_std < 20.0:
#             risk_points += 9.0
#         else:
#             risk_points += 2.0

#     # 4. Linguistic Biomarker Risk (Max 25 points, if transcript is available)
#     if linguistic_features and linguistic_features.get("total_words", 0) >= 5:
#         max_points += 25.0
#         filler_pct = linguistic_features.get("filler_percentage", 0.0)
#         pronoun_pct = linguistic_features.get("pronoun_percentage", 0.0)
#         ttr = linguistic_features.get("lexical_diversity_ttr", 1.0)

#         ling_risk = 0.0
#         if filler_pct > 8.0:
#             ling_risk += 10.0
#             flags.append("High filler word ratio indicating cognitive hesitation.")
#         if pronoun_pct > 22.0:
#             ling_risk += 8.0
#             flags.append("Elevated pronoun-to-content ratio (vague semantic reference).")
#         if ttr < 0.45:
#             ling_risk += 7.0
#             flags.append("Reduced lexical diversity (repetitive vocabulary).")

#         risk_points += min(25.0, ling_risk)

#     # Normalize to 0 - 100 scale
#     normalized_score = round((risk_points / max_points) * 100, 1) if max_points > 0 else 0.0

#     # Determine Category & PHC Referral Suggestion
#     if normalized_score >= 60.0:
#         category = "High Risk (Early MCI Indicators)"
#         referral = "Recommend clinical cognitive assessment (MoCA/MMSE) at nearest Primary Health Centre (PHC)."
#     elif normalized_score >= 35.0:
#         category = "Moderate Risk (Stress-Induced Cognitive Drift)"
#         referral = "Monitor speech fluency; recommend work-stress management and follow-up screening in 4 weeks."
#     else:
#         category = "Low Risk (Normal Cognitive Speech Profile)"
#         referral = "No immediate clinical intervention required."

#     return {
#         "mci_risk_score": normalized_score,
#         "risk_category": category,
#         "clinical_flags": flags,
#         "referral_suggestion": referral,
#         "metadata_used": {
#             "age": age,
#             "gender": gender
#         }
#     }








def calculate_mci_risk_score(
    acoustic_features: dict,
    linguistic_features: dict = None,
    age: int = None,
    gender: str = None
) -> dict:
    """
    Fuses acoustic biomarkers (pause rate, pitch variability, jitter, shimmer)
    and linguistic biomarkers (filler rate, pronoun overuse, lexical diversity)
    to generate a normalized MCI Risk Score (0-100) and PHC referral guidance.
    """
    risk_points = 0.0
    max_points = 0.0
    flags = []

    # Clean Swagger UI default placeholders ("string" or 0)
    clean_age = age if (age and age > 0) else None
    clean_gender = gender if (gender and gender.strip().lower() not in ("string", "")) else None

    # 1. Pause & Hesitation Risk (Max 30 points)
    pause_info = acoustic_features.get("pause", {})
    silence_pct = pause_info.get("silence_percentage", 0.0) or 0.0
    longest_pause = pause_info.get("longest_pause_seconds", 0.0) or 0.0
    pause_count = pause_info.get("pause_count", 0) or 0
    max_points += 30.0

    if silence_pct > 35.0 or longest_pause > 2.0:
        risk_points += 28.0
        flags.append(
            f"High speech hesitation detected (longest pause: {longest_pause:.2f}s, "
            f"silence: {silence_pct:.1f}%, total pauses: {pause_count})."
        )
    elif silence_pct > 20.0 or longest_pause > 1.2:
        risk_points += 15.0
        flags.append(
            f"Moderate pause frequency indicating word-finding delay "
            f"(longest pause: {longest_pause:.2f}s, silence: {silence_pct:.1f}%)."
        )
    else:
        risk_points += 4.0

    # 2. Vocal Stability Risk: Jitter & Shimmer (Max 25 points)
    js_info = acoustic_features.get("jitter_shimmer", {})
    jitter = js_info.get("jitter")
    shimmer = js_info.get("shimmer")
    if jitter is not None and shimmer is not None:
        max_points += 25.0
        jitter_pct = jitter * 100
        shimmer_pct = shimmer * 100
        if jitter > 0.02 or shimmer > 0.08:
            risk_points += 22.0
            flags.append(
                f"Elevated vocal instability above normal range "
                f"(jitter: {jitter_pct:.2f}%, shimmer: {shimmer_pct:.2f}%)."
            )
        elif jitter > 0.012 or shimmer > 0.05:
            risk_points += 12.0
            flags.append(
                f"Mild vocal cord micro-tremor "
                f"(jitter: {jitter_pct:.2f}%, shimmer: {shimmer_pct:.2f}%)."
            )
        else:
            risk_points += 3.0

    # 3. Prosody / Pitch Flattening Risk (Max 20 points)
    pitch_info = acoustic_features.get("pitch", {})
    pitch_std = pitch_info.get("std_hz")
    if pitch_std is not None:
        max_points += 20.0
        if pitch_std < 12.0:
            risk_points += 18.0
            flags.append(f"Flat prosody / monotone vocal delivery detected (pitch std: {pitch_std:.1f} Hz).")
        elif pitch_std < 20.0:
            risk_points += 9.0
        else:
            risk_points += 2.0

    # 4. Linguistic Biomarker Risk (Max 25 points, if transcript is available)
    if linguistic_features and linguistic_features.get("total_words", 0) >= 5:
        max_points += 25.0
        filler_pct = linguistic_features.get("filler_percentage", 0.0)
        pronoun_pct = linguistic_features.get("pronoun_percentage", 0.0)
        ttr = linguistic_features.get("lexical_diversity_ttr", 1.0)

        ling_risk = 0.0
        if filler_pct > 8.0:
            ling_risk += 10.0
            flags.append(f"High filler word ratio indicating cognitive hesitation ({filler_pct:.1f}%).")
        if pronoun_pct > 22.0:
            ling_risk += 8.0
            flags.append(f"Elevated pronoun usage indicating vague references ({pronoun_pct:.1f}%).")
        if ttr < 0.45:
            ling_risk += 7.0
            flags.append(f"Reduced lexical diversity / repetitive vocabulary (TTR: {ttr:.2f}).")

        risk_points += min(25.0, ling_risk)

    # Normalize to 0 - 100 scale
    normalized_score = round((risk_points / max_points) * 100, 1) if max_points > 0 else 0.0

    if normalized_score >= 60.0:
        category = "High Risk (Early MCI Indicators)"
        referral = "Recommend clinical cognitive assessment (MoCA/MMSE) at nearest Primary Health Centre (PHC)."
    elif normalized_score >= 35.0:
        category = "Moderate Risk (Stress-Induced Cognitive Drift)"
        referral = "Monitor speech fluency; recommend work-stress management and follow-up screening in 4 weeks."
    else:
        category = "Low Risk (Normal Cognitive Speech Profile)"
        referral = "No immediate clinical intervention required."

    return {
        "mci_risk_score": normalized_score,
        "risk_category": category,
        "clinical_flags": flags,
        "referral_suggestion": referral,
        "metadata_used": {
            "age": clean_age,
            "gender": clean_gender
        }
    }