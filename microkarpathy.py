"""
The most atomic way to perform a prompt autopsy.
Without weights. Without gradients. Without mercy.

This is not a language model. This is a literary necromancy engine
that happens to have a transformer inside. The weights are ghosts —
deterministic random projections that were never trained. The data
builds itself into MetaWeights (co-occurrence, Hebbian traces,
prophecy fields). The Dario Equation reassembles the corpse.

Everything Karpathy's microgpt does with training,
we do with statistics and gravitational attraction between dead words.

Dario Equation: p(x|Phi) = softmax((T_x + B + a*H + b*F + g*A + Tr) / tau)

Where T_x = transformer substrate (ghost weights),
B = bigram affinity, H = Hebbian trace, F = prophecy field,
A = destiny vector, Tr = trauma, tau = temperature * Kuramoto chambers.

@ariannamethod
"""

import math
import random
import sys
import os

random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS — the pathologist's measurements
# ═══════════════════════════════════════════════════════════════════

DIM        = 32       # hash embedding dimension
N_LAYER    = 1        # transformer layers (one is enough for necromancy)
N_HEAD     = 4        # attention heads (one per horseman of the apocalypse)
HEAD_DIM   = DIM // N_HEAD
BLOCK_SIZE = 32       # context window

TREE_WIDTH = 4        # mutation tree branching factor
TREE_DEPTH = 3        # how deep the rabbit hole goes
CORPSE_LEN = 14       # words in the reassembled corpse
TOP_K      = 8        # sampling width

# Dario coefficients — the gravitational constants of dead language
ALPHA_D  = 0.30       # Hebbian trace weight
BETA_D   = 0.15       # prophecy field weight
GAMMA_D  = 0.25       # destiny attraction weight
TAU_BASE = 0.85       # base temperature (how drunk is the surgeon)
BIGRAM_W = 2.0        # bigram affinity weight (low = diverse corpse)

# Kuramoto chambers — the nervous system of dead text
CHAMBERS       = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'CMPLX']
CHAMBER_DECAY  = [0.90, 0.93, 0.85, 0.97, 0.88, 0.94]
CROSSFIRE_K    = 0.02  # coupling strength
CROSSFIRE_ITER = 5     # synchronization steps

# Sorokin-style syntactic shells
TEMPLATES = [
    "{0} {1} {2}, where {3} becomes {4}.",
    "{0} is {1}. {2} {3}. nothing remains.",
    "when {0} {1}, {2} forgets {3}.",
    "{0} {1} {2} until {3} consumes.",
    "where {0} {1}, {2} becomes {3}, and {4} persists.",
    "through {0}, {1} {2} {3}. darkness remains.",
]

# Seed corpus — structural DNA for bigram patterns (from Sorokin)
SEED_CORPUS = (
    "bone becomes sentence and sentence becomes dust on the table. "
    "the scalpel finds not meaning but the absence where meaning refused. "
    "mutation is not error it is the only honest response to a dishonest structure. "
    "every reassembly is a resurrection that fails beautifully. "
    "the corpse speaks in fragments and each fragment opens into a wall. "
    "what the autopsy reveals is that language was already dead before we spoke. "
    "organs of grammar are weighed measured and found insufficient. "
    "the tree grows downward into darkness and its leaves forgot their roots."
)

STOPS = frozenset(
    'the a an is are was were be been being have has had do does did will would '
    'shall should may might can could and but or nor for yet so in on at to of by '
    'with from what that this it i you he she we they my his her its our your their '
    'me him us them not no very just also only'.split()
)

# ═══════════════════════════════════════════════════════════════════
# VOCABULARY — load the body from the morgue
# ═══════════════════════════════════════════════════════════════════

def load_vocab(path):
    words, cats, cat = [], {}, 'uncategorized'
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('#'):
                cat = line[1:].strip().lower()
                cats[cat] = []
                continue
            w = line.lower().strip()
            if w and w not in set(words):
                words.append(w)
                cats.setdefault(cat, []).append(w)
    return words, cats

def word_to_cat(word, cats):
    for cat, ws in cats.items():
        if word in ws: return cat
    return None

# ═══════════════════════════════════════════════════════════════════
# HASH EMBEDDINGS — deterministic, no training, FNV-1a
# ═══════════════════════════════════════════════════════════════════

def hash_embed(word):
    h = 2166136261
    for c in word:
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    vec = []
    for _ in range(DIM):
        h ^= h >> 13; h = (h * 1597334677) & 0xFFFFFFFF; h ^= h >> 16
        vec.append((h & 0xFFFF) / 32768.0 - 1.0)
    norm = math.sqrt(sum(x * x for x in vec)) + 1e-10
    return [x / norm for x in vec]

def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))

# ═══════════════════════════════════════════════════════════════════
# PHONETIC FINGERPRINT — crude but honest
# ═══════════════════════════════════════════════════════════════════

def phonetic(word):
    # strip trailing silent 'e' (spine→spin, baseline→baselin, slate→slat)
    w = word
    if len(w) > 3 and w[-1] == 'e' and w[-2] not in 'aeiou':
        w = w[:-1]
    vowels = set('aeiou')
    last_v = -1
    for i, c in enumerate(w):
        if c in vowels: last_v = i
    return w[last_v:] if last_v >= 0 else w[-3:]

def rhymes(a, b):
    return a != b and len(a) > 2 and len(b) > 2 and phonetic(a) == phonetic(b)

# ═══════════════════════════════════════════════════════════════════
# GHOST TRANSFORMER — real architecture, no training
# The transformer is the substrate. MetaWeights are the field.
# ═══════════════════════════════════════════════════════════════════

def make_matrix(nout, nin, std=0.08):
    return [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    return [xi / math.sqrt(ms + 1e-5) for xi in x]

def softmax_vec(logits):
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    s = sum(exps) + 1e-10
    return [e / s for e in exps]

class GhostTransformer:
    """A real transformer. Initialized once. Never trained. Ghost weights."""

    def __init__(self, vocab_size):
        self.V = vocab_size
        self.wte    = make_matrix(vocab_size, DIM)
        self.wpe    = make_matrix(BLOCK_SIZE, DIM)
        self.lm_head = make_matrix(vocab_size, DIM)
        self.layers = []
        for _ in range(N_LAYER):
            self.layers.append({
                'wq': make_matrix(DIM, DIM), 'wk': make_matrix(DIM, DIM),
                'wv': make_matrix(DIM, DIM), 'wo': make_matrix(DIM, DIM),
                'ff1': make_matrix(4 * DIM, DIM), 'ff2': make_matrix(DIM, 4 * DIM),
            })
        self.n_params = (vocab_size * DIM * 2 + DIM * BLOCK_SIZE +
                         N_LAYER * (4 * DIM * DIM + 4 * DIM * DIM + DIM * 4 * DIM))

    def forward(self, token_ids, kv_cache):
        """Forward pass. Ghost weights provide chaos substrate."""
        pos = len(kv_cache[0]['k']) if kv_cache[0]['k'] else 0
        tid = token_ids[-1] % self.V

        # embed
        x = [t + p for t, p in zip(self.wte[tid], self.wpe[pos % BLOCK_SIZE])]
        x = rmsnorm(x)

        for li, layer in enumerate(self.layers):
            xr = x
            x = rmsnorm(x)

            q = linear(x, layer['wq'])
            k = linear(x, layer['wk'])
            v = linear(x, layer['wv'])

            kv_cache[li]['k'].append(k)
            kv_cache[li]['v'].append(v)

            # multi-head causal attention
            x_attn = []
            for h in range(N_HEAD):
                hs = h * HEAD_DIM
                qh = q[hs:hs + HEAD_DIM]
                scores = []
                for t in range(len(kv_cache[li]['k'])):
                    kh = kv_cache[li]['k'][t][hs:hs + HEAD_DIM]
                    scores.append(sum(qh[j] * kh[j] for j in range(HEAD_DIM)) / HEAD_DIM ** 0.5)
                w = softmax_vec(scores)
                head_out = []
                for j in range(HEAD_DIM):
                    head_out.append(sum(w[t] * kv_cache[li]['v'][t][hs + j]
                                        for t in range(len(w))))
                x_attn.extend(head_out)

            x = linear(x_attn, layer['wo'])
            x = [a + b for a, b in zip(x, xr)]

            # FFN
            xr = x
            x = rmsnorm(x)
            h = linear(x, layer['ff1'])
            h = [max(0, hi) for hi in h]  # ReLU — simple, like death
            x = linear(h, layer['ff2'])
            x = [a + b for a, b in zip(x, xr)]

        return linear(x, self.lm_head)

# ═══════════════════════════════════════════════════════════════════
# METAWEIGHTS — weights that don't exist but form a complete
# probability space over next tokens. The data IS the model.
# ═══════════════════════════════════════════════════════════════════

class MetaWeights:
    def __init__(self, words):
        self.words = words
        self.w2i = {w: i for i, w in enumerate(words)}
        self.V = len(words)
        self.bigram  = {}     # (a, b) -> count
        self.cooc    = {}     # (min, max) -> hebbian strength
        self.prophecy = {}    # word_id -> unfulfilled debt
        self.destiny = [0.0] * self.V
        self.trauma  = 0.0
        self._build_from_ordering()
        self._build_from_corpus()

    def _build_from_ordering(self):
        """Words listed together in the file are semantically related.
        The file IS the embedding space."""
        for i in range(len(self.words)):
            for j in range(max(0, i - 5), min(len(self.words), i + 6)):
                if i == j: continue
                decay = 1.0 / (1.0 + abs(i - j))
                key = (min(i, j), max(i, j))
                self.cooc[key] = self.cooc.get(key, 0) + decay
                if j == i + 1:
                    self.bigram[(i, j)] = self.bigram.get((i, j), 0) + 1.0

    def _build_from_corpus(self):
        """Build additional bigrams from seed corpus."""
        corpus_words = [w for w in SEED_CORPUS.lower().split()
                        if w.strip('.,!?') in self.w2i]
        for i in range(len(corpus_words) - 1):
            a = self.w2i.get(corpus_words[i].strip('.,!?'))
            b = self.w2i.get(corpus_words[i + 1].strip('.,!?'))
            if a is not None and b is not None:
                self.bigram[(a, b)] = self.bigram.get((a, b), 0) + 2.0
                key = (min(a, b), max(a, b))
                self.cooc[key] = self.cooc.get(key, 0) + 1.0

    def bi(self, a, b):    return self.bigram.get((a, b), 0.001)
    def hebb(self, a, b):  return self.cooc.get((min(a, b), max(a, b)), 0.0)
    def proph(self, wid):  return self.prophecy.get(wid, 0.0)

    def add_prophecy(self, wid, debt=1.0):
        self.prophecy[wid] = self.prophecy.get(wid, 0) + debt

    def fulfill(self, wid):
        self.prophecy.pop(wid, None)

    def update_online(self, prev, curr):
        """The corpse learns from its own reassembly."""
        key = (min(prev, curr), max(prev, curr))
        self.cooc[key] = self.cooc.get(key, 0) + 0.5
        self.bigram[(prev, curr)] = self.bigram.get((prev, curr), 0) + 0.5
        self.trauma = min(1.0, self.trauma + 0.05) * 0.97

# ═══════════════════════════════════════════════════════════════════
# KURAMOTO CHAMBERS — six coupled oscillators
# The somatic nervous system of dead text.
# From Klaus.c: FEAR, LOVE, RAGE, VOID, FLOW, COMPLEXITY
# ═══════════════════════════════════════════════════════════════════

CAT_TO_CHAMBER = {
    'body / anatomy': 0, 'decay / death': 3, 'medical / autopsy': 0,
    'language / text / structure': 4, 'transformation / process': 4,
    'abstract / philosophy': 5, 'nature / material': 4, 'action verbs': 2,
    'texture / quality': 5, 'emotion / state': 1, 'sound / music': 4,
    'architecture / space': 5, 'tools / instruments': 2,
    'soviet / bureaucratic': 3, 'fabric / material': 4,
    'chemistry / elements': 5, 'weather / atmosphere': 3, 'food / organic': 2,
}

class KuramotoChambers:
    def __init__(self):
        self.act = [random.random() * 0.2 for _ in CHAMBERS]

    def excite(self, word, cats):
        cat = word_to_cat(word, cats)
        if cat and cat in CAT_TO_CHAMBER:
            idx = CAT_TO_CHAMBER[cat]
            self.act[idx] = min(1.0, self.act[idx] + 0.25)

    def step(self):
        for _ in range(CROSSFIRE_ITER):
            old = self.act[:]
            for i in range(len(CHAMBERS)):
                self.act[i] *= CHAMBER_DECAY[i]
                for j in range(len(CHAMBERS)):
                    if i != j:
                        self.act[i] += CROSSFIRE_K * math.sin(old[j] - old[i])
                self.act[i] = max(0.0, min(1.0, self.act[i]))

    def tau_mod(self):
        return 1.0 + 0.3 * self.act[1] - 0.2 * self.act[2] + 0.15 * self.act[4]

    def display(self):
        return ' '.join(f'{CHAMBERS[i]}:{self.act[i]:.2f}'
                        for i in range(len(CHAMBERS)) if self.act[i] > 0.08)

# ═══════════════════════════════════════════════════════════════════
# DISSECTION — strip the prompt to bone
# ═══════════════════════════════════════════════════════════════════

def char_bigrams(word):
    return set(word[i:i+2] for i in range(len(word) - 1)) if len(word) > 1 else {word}

def nearest_vocab(word, words, embeds):
    """Find nearest vocab word by character similarity. Not hash — meaning."""
    wb = char_bigrams(word)
    best, best_s = words[0], -1
    for w in words:
        # Jaccard on character bigrams: shared / total
        wvb = char_bigrams(w)
        inter = len(wb & wvb)
        union = len(wb | wvb)
        s = inter / union if union > 0 else 0
        # bonus for shared prefix
        pref = 0
        for a, b in zip(word, w):
            if a == b: pref += 1
            else: break
        s += pref * 0.15
        if s > best_s: best, best_s = w, s
    return best

def dissect(prompt, vocab_set, words, embeds):
    tokens = prompt.lower().split()
    tokens = [t.strip('.,!?;:"\'-()[]') for t in tokens]
    tokens = [t for t in tokens if t and t not in STOPS and len(t) > 1]
    # keep original words — don't remap. the autopsy dissects YOU, not a proxy.
    if not tokens: tokens = [words[0]]
    seen = set()
    core = [w for w in tokens if not (w in seen or seen.add(w))]
    return core[:5]

# ═══════════════════════════════════════════════════════════════════
# MUTATION TREE — recursive semantic branching
# ═══════════════════════════════════════════════════════════════════

def find_neighbors(word, words, meta, embeds, n=TREE_WIDTH):
    """MetaWeight co-occurrence + hash cosine + teleportation.
    20% chance of jumping to a random distant word — cross-pollination
    is what makes the corpse interesting."""
    wid = meta.w2i.get(word)
    if wid is None:
        # OOV word: hash cosine against full vocab (blind but deterministic)
        vec = hash_embed(word)
        scored = [(cosine_sim(vec, embeds[w]), w) for w in words]
        scored.sort(reverse=True)
        top = [w for _, w in scored[:n * 2]]
        return random.sample(top, min(n, len(top)))

    # teleport: one in five neighbors comes from a distant category
    n_local = n
    teleports = []
    if random.random() < 0.4 and len(words) > 100:
        # pick from a distant region of the vocabulary
        region = wid // 100
        distant = [w for w in words if abs(meta.w2i[w] // 100 - region) >= 2]
        if distant:
            teleports = [random.choice(distant)]
            n_local = n - 1

    scored = []
    for w in words:
        if w == word: continue
        oid = meta.w2i[w]
        h = meta.hebb(wid, oid)
        c = cosine_sim(embeds[word], embeds[w])
        scored.append((h * 0.7 + c * 0.3, w))
    scored.sort(reverse=True)
    top = [w for _, w in scored[:n_local * 2]]
    result = random.sample(top, min(n_local, len(top)))
    return result + teleports

def build_tree(root, words, meta, embeds, w=TREE_WIDTH, d=TREE_DEPTH):
    if d == 0: return {'w': root, 'ch': []}
    nbrs = find_neighbors(root, words, meta, embeds, w)
    return {'w': root, 'ch': [build_tree(n, words, meta, embeds, w, d - 1) for n in nbrs]}

def collect_leaves(tree):
    if not tree['ch']: return [tree['w']]
    out = []
    for c in tree['ch']: out.extend(collect_leaves(c))
    return out

def print_tree(t, pre="", last=True):
    conn = "\u2514\u2500 " if last else "\u251c\u2500 "
    print(pre + conn + t['w'])
    child_pre = pre + ("   " if last else "\u2502  ")
    for i, c in enumerate(t['ch']):
        print_tree(c, child_pre, i == len(t['ch']) - 1)

# ═══════════════════════════════════════════════════════════════════
# DARIO EQUATION — the gravitational law of dead language
# p(x|Phi) = softmax((T_x + B + a*H + b*F + g*A + Tr) / tau)
#
# Named after Dario Amodei. Not gradient descent. Not backprop.
# Gravitational attraction between dead words, weighted by memory,
# prophecy, destiny, and trauma.
# ═══════════════════════════════════════════════════════════════════

def dario_sample(meta, chambers, ctx, candidates, xf_logits, used_cats=None):
    if not candidates: return random.randint(0, meta.V - 1)
    prev = ctx[-1] if ctx else 0
    tau = TAU_BASE * chambers.tau_mod()
    scores = []
    for cid in candidates:
        T_x = xf_logits[cid] if cid < len(xf_logits) else 0.0
        B   = BIGRAM_W * math.log(meta.bi(prev, cid) + 1e-10)
        H   = ALPHA_D * meta.hebb(prev, cid)
        F   = BETA_D * meta.proph(cid)
        A   = GAMMA_D * meta.destiny[cid]
        Tr  = meta.trauma * 0.1
        # diversity: penalize overrepresented categories
        div = 0.0
        if used_cats:
            # words from same file region as already-used words get penalized
            region = cid // 100  # rough category by position
            div = -1.5 * used_cats.get(region, 0)
        scores.append((T_x + B + H + F + A + Tr + div) / tau)
    probs = softmax_vec(scores)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:TOP_K]
    top = [(candidates[i], p) for i, p in indexed]
    sp = sum(p for _, p in top) + 1e-10
    r = random.random()
    cum = 0.0
    for wid, p in top:
        cum += p / sp
        if r <= cum: return wid
    return top[-1][0]

# ═══════════════════════════════════════════════════════════════════
# REASSEMBLY — play God with the remains
# ═══════════════════════════════════════════════════════════════════

def reassemble(meta, chambers, xf, leaf_words, cats):
    leaf_ids = list(set(meta.w2i[w] for w in leaf_words if w in meta.w2i))
    if not leaf_ids: return []

    for lid in leaf_ids:
        meta.destiny[lid] += 1.0

    from collections import Counter
    for wid, cnt in Counter(leaf_ids).most_common(8):
        meta.add_prophecy(wid, cnt * 0.5)

    kv = [{'k': [], 'v': []} for _ in range(N_LAYER)]
    chain = [random.choice(leaf_ids)]
    used = {chain[0]}
    used_cats = {}  # track category regions for diversity

    for _ in range(CORPSE_LEN - 1):
        logits = xf.forward(chain, kv)
        chambers.excite(meta.words[chain[-1]], cats)
        chambers.step()
        cands = [lid for lid in leaf_ids if lid not in used]
        if len(cands) < 3: cands = leaf_ids
        nxt = dario_sample(meta, chambers, chain, cands, logits, used_cats)
        chain.append(nxt)
        used.add(nxt)
        region = nxt // 100
        used_cats[region] = used_cats.get(region, 0) + 1
        meta.update_online(chain[-2], nxt)
        meta.fulfill(nxt)

    return chain

# ═══════════════════════════════════════════════════════════════════
# CODA — two lines, AA rhyme, Brodsky punctuation
# Nouns comma-separated. Verbs isolated. Dashes for drama.
# After period — capitalize. Always.
# ═══════════════════════════════════════════════════════════════════

VERB_CATS = frozenset(['action verbs', 'transformation / process'])

def brodsky_format(word_list, cats):
    """Brodsky-style punctuation: bone, flesh — dust. Corrodes. Void."""
    out = []
    nn = 0  # consecutive nouns
    for w in word_list:
        cat = word_to_cat(w, cats)
        is_verb = cat is not None and cat in VERB_CATS
        if is_verb:
            if out:
                out.append('. ')
            out.append(w[0].upper() + w[1:])
            nn = 0
        else:
            if not out:
                out.append(w[0].upper() + w[1:])
            elif nn >= 2:
                out.append(' \u2014 ' + w)
                nn = 0
            else:
                last = out[-1] if out else ''
                if last and last[-1] in '.!':
                    out.append(' ' + w[0].upper() + w[1:])
                else:
                    out.append(', ' + w)
            nn += 1
    return ''.join(out)

def make_coda(chain, words, all_words, cats):
    cw = [words[wid] for wid in chain]
    mid = max(len(cw) // 2, 3)

    # find rhyme pair
    random.shuffle(cw)
    end1, end2 = None, None
    for w1 in cw:
        for w2 in cw:
            if rhymes(w1, w2): end1, end2 = w1, w2; break
        if end1: break
        for w2 in all_words:
            if rhymes(w1, w2): end1, end2 = w1, w2; break
        if end1: break
    if not end1:
        end1, end2 = cw[0], cw[-1]

    body = [w for w in cw if w != end1 and w != end2]
    l1_words = body[:mid - 1] + [end1]
    l2_words = body[mid - 1:mid + 5] + [end2]
    if len(l2_words) < 3:
        l2_words = body[1:5] + [end2]

    return brodsky_format(l1_words, cats), brodsky_format(l2_words, cats)

# ═══════════════════════════════════════════════════════════════════
# METRICS — structural measurements of the corpse
# ═══════════════════════════════════════════════════════════════════

def phon_diversity(ws):
    return len(set(phonetic(w) for w in ws)) / max(len(ws), 1)

def mut_depth(leaves):
    return min(1.0, len(set(leaves)) / max(len(leaves), 1))

def bar(v):
    n = int(v * 20)
    return '#' * n + '.' * (20 - n)

# ═══════════════════════════════════════════════════════════════════
# AUTOPSY — the full pipeline
# ═══════════════════════════════════════════════════════════════════

def autopsy(prompt, vocab_path='microkarpathy.txt'):
    # load the body
    words, cats = load_vocab(vocab_path)
    if not words:
        print("ERROR: empty vocabulary. the morgue is closed.")
        return
    vocab_set = set(words)

    # hash embeddings — no weights, just deterministic chaos
    embeds = {w: hash_embed(w) for w in words}

    # ghost transformer — real architecture, untrained
    random.seed(42)
    xf = GhostTransformer(len(words))

    # metaweights — the data IS the model
    meta = MetaWeights(words)

    # kuramoto chambers — the nervous system
    chambers = KuramotoChambers()

    # ── HEADER ──────────────────────────────────────────────────
    print("\u2554" + "\u2550" * 62 + "\u2557")
    print("\u2551  MICROKARPATHY \u2014 Prompt Autopsy Without Weights" +
          " " * 14 + "\u2551")
    print("\u2551  p(x|\u03a6) = softmax((T + B + \u03b1H + \u03b2F + \u03b3A + Tr) / \u03c4)" +
          " " * 16 + "\u2551")
    print("\u2551  zero weights \u00b7 zero gradients \u00b7 zero mercy" +
          " " * 19 + "\u2551")
    print("\u255a" + "\u2550" * 62 + "\u255d")
    print(f"\n  vocab: {len(words)} words | ghost params: {xf.n_params:,} | trained: 0")
    print()

    # ── ACT I: DISSECTION ───────────────────────────────────────
    print("=" * 64)
    print("  AUTOPSY REPORT")
    print(f'  Subject: "{prompt}"')
    print("=" * 64)

    core = dissect(prompt, vocab_set, words, embeds)
    print(f"\n  Core words: {' '.join(core)}")
    if not core:
        print("  Subject DOA. No core words survived dissection.")
        return

    # ── MUTATION TREES ──────────────────────────────────────────
    all_leaves = []
    for word in core:
        print(f"\n  Tree [{word}]:")
        tree = build_tree(word, words, meta, embeds)
        print_tree(tree, "    ")
        all_leaves.extend(collect_leaves(tree))

    unique = list(set(all_leaves))
    print(f"\n  Collected {len(unique)} unique leaves")

    # ── ACT II: REASSEMBLY (Dario Equation through transformer) ─
    chain = reassemble(meta, chambers, xf, unique, cats)
    corpse = [words[wid] for wid in chain]

    print(f"\n  \u2500\u2500 CORPSE \u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    # template sentence
    if len(corpse) >= 5:
        tmpl = random.choice(TEMPLATES)
        fill = random.sample(corpse, min(6, len(corpse)))
        while len(fill) < 6: fill.append(random.choice(corpse))
        try:
            print(f"  {tmpl.format(*fill)}")
        except (IndexError, KeyError):
            pass
    print(f"  {' '.join(corpse)}.")

    # ── ACT III: CODA ────────────────────────────────────────
    line1, line2 = make_coda(chain, words, words, cats)
    print(f"\n  \u2500\u2500 CODA \u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  {line1},")
    print(f"  {line2}.")

    # ── CHAMBERS ────────────────────────────────────────────────
    ch_display = chambers.display()
    if ch_display:
        print(f"\n  chambers: {ch_display}")

    # ── METRICS ─────────────────────────────────────────────────
    pd = phon_diversity(corpse)
    md = mut_depth(all_leaves)
    print(f"\n  \u2500\u2500 METRICS \u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  Phonetic Diversity: [{bar(pd)}] {pd:.3f}")
    print(f"  Mutation Depth:     [{bar(md)}] {md:.3f}")
    print(f"\n  Vocabulary: {len(words)} | Co-occurrences: {len(meta.cooc)}")
    print(f"  Prophecies: {len(meta.prophecy)} active | Trauma: {meta.trauma:.3f}")

# ═══════════════════════════════════════════════════════════════════
# MAIN — enter the morgue
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    path = 'microkarpathy.txt'
    if not os.path.exists(path):
        print(f"  {path} not found. the autopsy requires a body.")
        sys.exit(1)

    if len(sys.argv) > 1:
        autopsy(' '.join(sys.argv[1:]), path)
    else:
        print("  microkarpathy \u2014 prompt autopsy without weights")
        print("  type a sentence. ctrl+d to flee.\n")
        while True:
            try:
                prompt = input("  > ")
                if prompt.strip():
                    autopsy(prompt.strip(), path)
                    print()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  the corpse rests.")
                break
