# helpers for outlet side detection and speaker-side cues


def side_from_L(L):
    if L is None:
        L = 0.0
    elif L > 2.5:
        return "right"
    elif L < -2.5:
        return "left"
    else:
        return "center"


LEFT_CUES = ["democrat", "voting-rights", "labor", "civil rights", "progressive"]
RIGHT_CUES = [
    "republican",
    "gop",
    "business association",
    "chamber of commerce",
    "conservative",
    "gun rights",
]


def is_left_cue(text):
    if text is None:
        text = ""
    for cue in LEFT_CUES:
        if cue in text.lower():
            return True
    return False


def is_right_cue(text):
    for cue in RIGHT_CUES:
        if cue in text.lower():
            return True
    return False


def speaker_side(speaker_name):
    if speaker_name is None:
        speaker_name = ""
    if is_left_cue(speaker_name):
        return "left"
    elif is_right_cue(speaker_name):
        return "right"
    else:
        return "unknown"
