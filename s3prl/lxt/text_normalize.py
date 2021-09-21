import re

def pad_space(utterance_text: str):
    return " " + utterance_text + " "

def asr_norm(utterance_text: str):
    original_text = utterance_text
    utterance_text = utterance_text.strip()
    utterance_text = re.sub(r"\%", " percent ", utterance_text).strip()
    utterance_text = re.sub(r"\[ *citation needed *\]", " ", utterance_text).strip()
    utterance_text = re.sub(r" no\. ", " number ", pad_space(utterance_text)).strip()

    # Remove known symbols
    chars = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,', ".split(",")
    utterance_text = utterance_text.upper()
    new_chars = []
    for char in utterance_text:
        if char not in chars:
            new_chars.append(char)
    for char in new_chars:
        utterance_text = utterance_text.replace(char, " ")
    
    utterance_text = re.sub(r" CNN ", " C N N ", pad_space(utterance_text)).strip()
    utterance_text = re.sub(r" CNN'S ", " C N N'S ", pad_space(utterance_text)).strip()
    utterance_text = re.sub(r" KM ", " KILOMETERS ", pad_space(utterance_text)).strip()
    utterance_text = re.sub(r" DEGC ", " DEGREES CELSIUS ", pad_space(utterance_text)).strip()
    utterance_text = re.sub(r" DEGF ", " DEGREES FAHRENHEIT ", pad_space(utterance_text)).strip()
    utterance_text = re.sub(r" +N'T", "N'T", utterance_text)
    utterance_text = re.sub(r" +", " ", utterance_text)
    utterance_text = re.sub(r" +'", "'", utterance_text)
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
