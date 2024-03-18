from collections import Counter

def get_pairs(word):
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i+1]))
    return pairs

def bpe_tokenization(vocab, text):
    while True:
        pairs = get_pairs(text)
        freq = Counter(pairs)
        max_freq_pair = max(freq, key=freq.get)
        if max_freq_pair not in vocab:
            break
        new_token = ''.join(max_freq_pair)
        text = text.replace(''.join(max_freq_pair), new_token)
        vocab.add(max_freq_pair)
    print(pairs)
    return text

vocab = set()
text = "hello world"
tokens = bpe_tokenization(vocab, text)
print(tokens)
