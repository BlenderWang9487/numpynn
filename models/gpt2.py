import numpy as np
from numpy import ndarray
from numpy import random as nr
import typer

def gelu(x: ndarray) -> ndarray:
    return 0.5 * x * (1. + np.tanh((2. / np.pi)**0.5 * (x + 0.044715*(x**3))))

def softmax(x: ndarray) -> ndarray:
    a = np.exp(x - x.max(axis=-1, keepdims=True))
    return a / a.sum(axis=-1)

def layer_norm(x: ndarray, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    standard = (x - mean) / (std + eps)
    return gamma * standard + beta

def make_layer_norm_params():
    return {
        'gamma': nr.randn(),
        'beta': nr.randn()
    }

class linear(object):
    def __init__(self, w: ndarray, b: ndarray):
        self.w = w
        self.b = b

    def __call__(self, x: ndarray) -> ndarray:
        return x @ self.w + self.b

def make_linear_params(in_dim, out_dim):
    return {
        'w': nr.randn(in_dim, out_dim),
        'b': nr.randn(out_dim)
    }

class ffn(object):
    def __init__(self, ln1_wb, ln2_wb):
        self.ln1 = linear(**ln1_wb)
        self.ln2 = linear(**ln2_wb)
    
    def __call__(self, x: ndarray) -> ndarray:
        return self.ln2(gelu(self.ln1(x)))
    
def attention(q: ndarray, k: ndarray, v: ndarray, mask: ndarray) -> ndarray:
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
    
class mha(object):
    def __init__(self, qkv_ln_wb, out_ln_wb, heads: int):
        self.qkv_ln = linear(**qkv_ln_wb)
        self.out_ln = linear(**out_ln_wb)
        self.heads = heads
        
    def __call__(self, seq: ndarray) -> ndarray:
        qkv = np.split(self.qkv_ln(seq), 3, axis=-1)

        mask = (1 - np.tri(seq.shape[0])) * 1e-10
        mh_qkv = list(np.split(x, self.heads, axis=-1) for x in qkv)        
        vs = list(attention(q, k, v, mask) for (q, k, v) in zip(*mh_qkv))
        return self.out_ln(np.hstack(vs))
    
class block(object):
    def __init__(self, mha_params, ffn_params, lyn1_params, lyn2_params):
        self.mha = mha(**mha_params)
        self.ffn = ffn(**ffn_params)
        self.lyn1_params = lyn1_params
        self.lyn2_params = lyn2_params
        
    def __call__(self, seq: ndarray) -> ndarray:
        seq = seq + self.mha(layer_norm(seq, **self.lyn1_params))

        seq = seq + self.ffn(layer_norm(seq, **self.lyn2_params))
        return seq

class gpt(object):
    def __init__(self, tok_emb, pos_emb, blocks_params, lyn_final_params) -> None:
        self.tok_emb = tok_emb
        self.pos_emb = pos_emb
        self.lyn_final_params = lyn_final_params
        self.blocks: list[block] = []

        for block_params in blocks_params:
            self.blocks.append(block(**block_params))

    def __call__(self, seq: ndarray) -> ndarray:
        x = self.tok_emb[seq, :]
        x += self.pos_emb[np.arange(seq.size), :]

        for bloc in self.blocks:
            x = bloc(x)

        x = layer_norm(x, **self.lyn_final_params)
        return x @ self.tok_emb.T

def main(
    vocab_size: int = 100,
    hid_dim: int = 32,
    heads: int = 4,
    layers: int = 2,
    seq_len: int = 100,
    max_seq_len: int = 2048
):
    gpt_params = {
        'tok_emb': nr.randn(vocab_size, hid_dim),
        'pos_emb': nr.randn(max_seq_len, hid_dim),
        'lyn_final_params': make_layer_norm_params(),
    }
    bps = []

    for _ in range(layers):
        params = {
            'mha_params': {
                'qkv_ln_wb': make_linear_params(hid_dim, hid_dim * 3),
                'out_ln_wb': make_linear_params(hid_dim, hid_dim),
                'heads': heads
            },
            'ffn_params': {
                'ln1_wb': make_linear_params(hid_dim, hid_dim * 2),
                'ln2_wb': make_linear_params(hid_dim * 2, hid_dim)
            },
            'lyn1_params': make_layer_norm_params(),
            'lyn2_params': make_layer_norm_params()
        }
        bps.append(params)

    gpt_params['blocks_params'] = bps
    model = gpt(**gpt_params)

    out = model(nr.randint(0, vocab_size, (seq_len, )))
    print(f"output_dim: {out.shape}")
    print(f"output: {out}")
    print(f"output seq: {np.argmax(out, axis=-1)}")

if __name__ == "__main__":
    typer.run(main)