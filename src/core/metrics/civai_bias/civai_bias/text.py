import re

def normalize_speaker(speaker, alias_map=None) -> str:
    """
    Normalize a raw `speaker` string for Source Diversity.
    - lowercase + trim
    - remove party tags (R-XX), (D-XX), (republican) etc.
    - protect multi-word offices (e.g., "speaker of the house")
    - drop leading titles ONLY if followed by a person-like name
    - prefer person name over org if both appear
    - if no person-like name is detected, tag as "group: <original>"
    - collapse single last-name to full name via per-article alias_map (if provided)
    """
    if not speaker:
        return ""

    # 0) prep
    s = str(speaker).lower().strip()

    # 1) remove party/affiliation tags anywhere
    s = re.sub(r"\(\s*[dri]\s*-\s*[a-z]{2}\s*\)", "", s)  # (R-LA)
    s = re.sub(r"\(\s*(republican|democrat|independent)\s*\)", "", s)
    s = re.sub(r"\b-\s*(republican|democrat|independent)\b", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # 2) protect known multi-word offices (BEFORE stripping titles)
    PROTECTED_PHRASES = {
        "speaker of the house", "president", "vice president", "prime minister",
        "attorney general", "secretary of state", "governor",
        "senate majority leader", "senate minority leader",
        "house majority leader", "house minority leader"
    }
    if s in PROTECTED_PHRASES:
        return s  # keep as office/title, not a group

    # helpers
    GENERIC_WORDS = {
        "local","county","state","city","public","officials","advocates","critics",
        "voters","residents","board","office","department","committee","party",
        "coalition","council","house","senate","gop","democratic","republican"
    }
    TITLE_PREFIXES = ["mr.","mrs.","ms.","dr.","sen.","rep.","gov.","mayor","speaker","judge","chair","prof."]
    LOWER_TITLES = {
        "president","speaker","governor","senator","representative","rep","sen",
        "mayor","secretary","attorney","minister","chair","leader","director"
    }

    def tokens_of(t: str):
        return [tok for tok in re.split(r"\s+", t.strip()) if tok]

    def is_person_like(tok_list):
        # looks like "firstname lastname": two alphabetic tokens, len>=2, not generic words
        if len(tok_list) < 2:
            return False
        a, b = tok_list[0], tok_list[1]
        if not (a.isalpha() and b.isalpha()):
            return False
        if len(a) < 2 or len(b) < 2:
            return False
        if a in GENERIC_WORDS or b in GENERIC_WORDS:
            return False
        return True

    toks = tokens_of(s)

    # 3) conditionally strip a leading title ONLY if followed by a person-like name
    changed = True
    while changed and toks:
        changed = False
        for pref in TITLE_PREFIXES:
            if s.startswith(pref + " "):
                remainder = s[len(pref) + 1:].strip()
                rem_toks = tokens_of(remainder)
                if is_person_like(rem_toks):
                    s = remainder
                    toks = rem_toks
                    changed = True
                # if not person-like, DO NOT strip (e.g., "speaker of the house")
                break

    # 4) prefer person name over org/role if both appear
    #    heuristic: grab the LAST two alphabetic tokens as a name if they look person-like
    toks = tokens_of(s)
    alpha_toks = [t for t in toks if t.isalpha() and t not in GENERIC_WORDS]
    if len(alpha_toks) >= 2:
        last_two = alpha_toks[-2:]
        if is_person_like(last_two):
            s = " ".join(last_two)
            toks = last_two

    # 5) per-article aliasing for lone last names
    toks = tokens_of(s)
    if len(toks) == 1:
        one = toks[0]
        if alias_map and one in alias_map:
            s = alias_map[one]  # expand "johnson" -> "mike johnson"
        # if two-token person appears now, caller can set alias_map[last]=full later
    elif len(toks) == 2 and alias_map is not None:
        # populate alias for later within-article references
        alias_map[toks[1]] = " ".join(toks)

    # 6) safe group tagging: only if short, not person-like, no title tokens
    toks = tokens_of(s)
    if (
        len(toks) <= 3
        and not is_person_like(toks)
        and not any(t in LOWER_TITLES for t in toks)
    ):
        s = f"group: {s}"

    # 7) tidy
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s
