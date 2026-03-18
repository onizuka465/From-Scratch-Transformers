import numpy as np

d_model=512

WQ = np.random.randn(d_model, d_model)
WK = np.random.randn(d_model, d_model)
WV = np.random.randn(d_model, d_model)

d_ff = 2048

W1 = np.random.randn(d_model,d_ff)
W2 = np.random.randn (d_ff, d_model)
b1 = np.zeros(d_ff)
b2 = np.zeros(d_model)

X = np.random.randn(1, 6, d_model)
y = np.random.randn(1, 6, d_model)

vocab_size= 10000
W_out = np.random.randn(d_model, vocab_size)

#funções

def softmax(x):
    e_x = np.exp(x)     
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def self_attention(x):
    Q = x @ WQ
    K = x @ WK
    V = x @ WV
    K_T = K.transpose (0, 2, 1)
    scores = Q @ K_T
    scores_scaled = scores / np.sqrt(d_model)
    weigths = softmax(scores_scaled)
    return weigths @ V

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean)/ np.sqrt(var + epsilon)

def ffn(x):
    camada1 = np.maximum(0, x @ W1 + b1)
    camada2 = camada1 @ W2 + b2
    return camada2

def EncoderBlock(X):
    X_att = self_attention(X)
    X_norm1 = layer_norm(X + X_att)
    X_ffn = ffn(X_norm1)
    X_out = layer_norm(X_norm1 + X_ffn)
    return X_out

def create_casual_mask (seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask[mask == 1] = -np.inf
    return mask

def cross_attention(encoder_output, decoder_state):
    Q = decoder_state @ WQ
    K = encoder_output @ WK
    V = encoder_output @ WV
    K_T = K.transpose (0, 2, 1)
    scores = Q @ K_T
    scores_scaled = scores / np.sqrt(d_model)
    weigths = softmax(scores_scaled)
    return weigths @ V

def Masked_SelfAttention(x):
    Q = x @ WQ
    K = x @ WK
    V = x @ WV
    K_T = K.transpose (0, 2, 1)
    scores = Q @ K_T
    scores_scaled = scores / np.sqrt(d_model)
    
    seq_len = x.shape[1]
    M = create_casual_mask(seq_len)
    scores_masked = scores_scaled   + M
    weights = softmax(scores_masked)
    return weights @ V
    
def DecoderBlock(y, Z):
    y_mask = Masked_SelfAttention(y)
    y_norm1 = layer_norm(y + y_mask)
    y_cross = cross_attention(y_norm1, Z)
    y_norm2 = layer_norm(y_cross + y_norm1)
    y_ffn = ffn(y_norm2)
    y_out = layer_norm(y_ffn + y_norm2)
    logits = y_out @ W_out
    probs =softmax(logits)
    return probs

encoder_input = np.random.randn(1, 2, d_model)
Z = encoder_input
for i in range(6):
    Z = EncoderBlock(Z)

decoder_input = np.random.randn (1, 1, d_model)

max_tokens = 20
contador = 0

while True:
    probs = DecoderBlock (decoder_input, Z)
    next_token = np.argmax (probs[0, -1, :])
    if next_token == 0:
        print ("Fim de geração")
        break
    novo_token = np.random.randn (1,1, d_model)
    decoder_input = np.concatenate([decoder_input, novo_token], axis=1)
    contador += 1

