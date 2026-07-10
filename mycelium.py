#!/usr/bin/env python3
"""
mycelium.py — microkarpathy's memory organ. Reference implementation (Python).

kk x llm-wiki, with no LLM and no trained weights:
  - from Karpathy's llm-wiki: ingest sources into a persistent, compounding store,
    then QUERY by unfolding a synthesis across everything ever eaten.
  - from Dario's kk_kernel: knowledge is not RETRIEVED, it RESONATES — each entry
    carries a metaweight fingerprint and the current field scores against it (X.Wr).
  - the empty slot kk left (embed_fn = NULL -> lexical fallback, kk_kernel.h:88-112)
    is filled by CHARACTER-N-GRAM HASHING: a zero-weight embedder that feels word
    resemblance in ANY script.

Dario injects gently, at sentence boundaries. The morgue does not. mycelium splices
resonant fragments from many books into one deranged corpse — cross-source injection
gone full wild. This is the reference; the browser (microkarpathy.html) is the engine.

  python3 mycelium.py ingest a.txt b.txt      # eat books into .mycelium
  python3 mycelium.py unfold "the network remembers the drowned"
  python3 mycelium.py demo                     # self-contained, ingests dario/docs
  python3 mycelium.py lint
"""
import sys, os, json, math, re

DIM = 96

# ── char n-gram hash embedding (cached) — subword resemblance, any script ──
_gram_cache = {}
_word_cache = {}

def _fnv_vec(s):
    v = _gram_cache.get(s)
    if v is not None:
        return v
    h = 2166136261
    for c in s:
        h ^= ord(c); h = (h * 16777619) & 0xFFFFFFFF
    v = []
    for _ in range(DIM):
        h ^= h >> 13; h = (h * 1597334677) & 0xFFFFFFFF; h ^= h >> 16
        v.append((h & 0xFFFF) / 32768.0 - 1.0)
    _gram_cache[s] = v
    return v

def embed(word, n=3):
    v = _word_cache.get(word)
    if v is not None:
        return v
    w = "^" + word + "$"
    grams = [w[i:i + n] for i in range(len(w) - n + 1)] or [w]
    vec = [0.0] * DIM
    for g in grams:
        gv = _fnv_vec(g)
        for i in range(DIM):
            vec[i] += gv[i]
    norm = math.sqrt(sum(x * x for x in vec)) + 1e-10
    vec = [x / norm for x in vec]
    _word_cache[word] = vec
    return vec

def mean_embed(tokens):
    vec = [0.0] * DIM
    n = 0
    for t in tokens:
        e = embed(t)
        for i in range(DIM):
            vec[i] += e[i]
        n += 1
    if n == 0:
        return vec
    norm = math.sqrt(sum(x * x for x in vec)) + 1e-10
    return [x / norm for x in vec]

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

# ── segmentation + tokens: eat any formatting ──
STOPS = frozenset((
    "the a an and or but of to in on at is are was were be been it its this that these "
    "those i you he she we they me him her us them my your his our their as by for with "
    "from into over under so not no do does did have has had will would can could he's "
    "it's there here then than when where what which who whom whose all any some more most"
).split())

def segment(text):
    entries = []
    for block in re.split(r'\n\s*\n', text):
        line = ' '.join(block.split())
        if not line:
            continue
        for s in re.split(r'(?<=[.!?])\s+', line):
            s = s.strip()
            if len(s) > 12:
                entries.append(s)
    return entries

def tokenize(entry):
    toks = [t.strip('.,!?;:"\'()[]{}—–-*_') for t in entry.lower().split()]
    return [t for t in toks if len(t) > 2 and t not in STOPS]

# ── the mycelium: compounding fingerprint store over everything eaten ──
class Mycelium:
    def __init__(self):
        self.entries = []       # {text, source, emb, toks}
        self.freq = {}          # token -> count
        self.sources = {}       # token -> sorted sources
        self.edges = {}         # "a\tb" (a<b) -> weight
        self.corpora = []       # ingested source names

    def _edge(self, a, b, w):
        if a == b:
            return
        k = a + "\t" + b if a < b else b + "\t" + a
        self.edges[k] = self.edges.get(k, 0.0) + w

    def ingest(self, text, source):
        if source not in self.corpora:
            self.corpora.append(source)
        for entry in segment(text):
            toks = tokenize(entry)
            if len(toks) < 2:
                continue
            # kk-style fingerprint: the entry's char-ngram metaweight vector
            self.entries.append({'text': entry, 'source': source,
                                 'emb': mean_embed(toks), 'toks': toks})
            for i, t in enumerate(toks):
                self.freq[t] = self.freq.get(t, 0) + 1
                s = set(self.sources.get(t, [])); s.add(source)
                self.sources[t] = sorted(s)
                if i > 0:
                    self._edge(toks[i - 1], t, 1.0)      # cooc (adjacency)
            uniq = list(dict.fromkeys(toks))
            for i in range(len(uniq)):
                for j in range(i + 1, min(i + 8, len(uniq))):
                    self._edge(uniq[i], uniq[j], 0.3)    # SPA (shared entry)

    def resonate(self, field_emb, k=8, per_source_cap=3):
        """kk's X.Wr: score every stored fingerprint against the current field.
        Knowledge does not get searched — it resonates. Capped per source so the
        corpse draws from MANY books, not one."""
        scored = sorted(
            ((cosine(field_emb, e['emb']), e) for e in self.entries),
            key=lambda x: -x[0])
        out, seen = [], {}
        for score, e in scored:
            if seen.get(e['source'], 0) >= per_source_cap:
                continue
            seen[e['source']] = seen.get(e['source'], 0) + 1
            out.append((score, e))
            if len(out) >= k:
                break
        return out

    def save(self, path=".mycelium"):
        json.dump({'entries': self.entries, 'freq': self.freq,
                   'sources': self.sources, 'edges': self.edges,
                   'corpora': self.corpora}, open(path, 'w'), ensure_ascii=False)

    @classmethod
    def load(cls, path=".mycelium"):
        m = cls()
        if os.path.exists(path):
            d = json.load(open(path))
            m.entries, m.freq = d['entries'], d['freq']
            m.sources, m.edges, m.corpora = d['sources'], d['edges'], d['corpora']
        return m

    def lint(self):
        before = len(self.edges)
        self.edges = {k: v * 0.9 for k, v in self.edges.items() if v * 0.9 > 0.15}
        strong = set()
        for k in self.edges:
            a, b = k.split("\t"); strong.add(a); strong.add(b)
        orphans = [t for t, f in self.freq.items() if f <= 1 and t not in strong]
        return before - len(self.edges), orphans

def _cut(text, lo=6, hi=16):
    """Cut a fragment to a clause — the morgue does not quote whole sentences."""
    words = text.split()
    n = lo + (len(words) % (hi - lo + 1))
    frag = ' '.join(words[:n]).rstrip('.,;:')
    return frag

def unfold(myc, prompt, k=9):
    """The wild injection: dissect the prompt into a field, resonate it against the
    whole compounded store, and splice the resonant fragments from DIFFERENT books
    into one corpse. Dario trickles at sentence boundaries; the morgue floods."""
    core = tokenize(prompt)
    field = mean_embed(core)
    print(f'  prompt : "{prompt}"')
    print(f'  field  : {core or "(none)"}')
    res = myc.resonate(field, k)
    if not res:
        print("  the mycelium is empty — ingest something first.")
        return 0
    print(f"\n  ── CORPSE (spliced from {len(myc.corpora)} corpora by resonance) ──")
    corpse = []
    touched = set()
    for score, e in res:
        touched.add(e['source'])
        corpse.append((_cut(e['text']), e['source'], round(score, 3)))
    # interleave as one deranged passage
    print("  " + ".  ".join(c for c, _, _ in corpse) + ".")
    print(f"\n  ── lineage (which book each clause bled from) ──")
    for frag, src, score in corpse:
        print(f"    [{src:<26} r={score}]  {frag[:60]}")
    print(f"\n  -> the corpse drew on {len(touched)}/{len(myc.corpora)} corpora "
          f"({sorted(touched)})")
    return len(touched)

# ── main ──
def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "demo"

    if cmd == "ingest":
        m = Mycelium.load()
        for path in args[1:]:
            if os.path.exists(path):
                m.ingest(open(path, encoding="utf-8", errors="ignore").read(),
                         os.path.basename(path))
                print(f"  ate {path}: {len(m.entries)} fragments, "
                      f"{len(m.freq)} tokens, {len(m.edges)} edges")
        m.save()

    elif cmd == "unfold":
        unfold(Mycelium.load(), " ".join(args[1:]))

    elif cmd == "lint":
        m = Mycelium.load()
        pruned, orphans = m.lint(); m.save()
        print(f"  lint: decayed {pruned} weak edges, {len(orphans)} orphans")

    elif cmd == "demo":
        import time
        docs = os.path.expanduser("~/arianna/dario/docs")
        books = ["mycorrhizal_networks.txt", "bioluminescence.txt",
                 "bach_counterpoint.txt", "polynesian_navigation.txt",
                 "byzantine_iconography.txt"]
        print("  mycelium demo — kk x llm-wiki, the morgue's memory organ\n")
        m = Mycelium()
        t0 = time.time()
        for b in books:
            p = os.path.join(docs, b)
            if os.path.exists(p):
                m.ingest(open(p, encoding="utf-8", errors="ignore").read(), b)
        dt = time.time() - t0
        print(f"  ate {len(m.corpora)} books: {len(m.entries)} fragments, "
              f"{len(m.freq)} tokens, {len(m.edges)} edges in {dt:.2f}s\n")
        # a deliberately cross-domain prompt — no single book owns it
        t1 = time.time()
        n = unfold(m, "the network remembers the light beneath the drowned world")
        print(f"\n  (resonate+unfold over {len(m.entries)} fragments: "
              f"{time.time()-t1:.3f}s)")
        # persistence round-trip
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mycelium_demo")
        m.save(tmp); r = Mycelium.load(tmp); os.remove(tmp)
        print(f"  persistence: reloaded {len(r.entries)} fragments, "
              f"{len(r.edges)} edges (equal={len(r.entries)==len(m.entries)})")

    else:
        print(__doc__)

if __name__ == "__main__":
    main()
