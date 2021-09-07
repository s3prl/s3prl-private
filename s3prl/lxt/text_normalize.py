import re

def asr_norm(utterance_text: str):
    chars = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,', ".split(",")
    utterance_text = utterance_text.upper()
    new_chars = []
    for char in utterance_text:
        if char not in chars:
            new_chars.append(char)
    for char in new_chars:
        utterance_text = utterance_text.replace(char, " ")

    utterance_text = re.sub(r" *' *", "' ", utterance_text)
    utterance_text = re.sub(r" +", " ", utterance_text)
    return utterance_text.strip()

def qbe_norm(utterance_text: str):
    chars = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z, ".split(",")
    utterance_text = utterance_text.upper()
    new_chars = []
    for char in utterance_text:
        if char not in chars:
            new_chars.append(char)
    for char in new_chars:
        utterance_text = utterance_text.replace(char, " ")

    utterance_text = re.sub(r" +", " ", utterance_text)
    return utterance_text.strip()
