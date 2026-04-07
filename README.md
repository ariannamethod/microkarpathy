# microKarpathy  
**by Arianna Method**  
---
  
microKarpathy is a Python script (atomic) which takes your innocent prompts like a psychotic linguist, tears them apart like a psychopathic pathologist, builds a recursive tree of semantic mutations, and then, like Frankenstein having a particularly creative day — reassembles the corpse into something *new*. for educational purposes only. 

no dependencies. no internet. no autoregressive generation. ain't no conscience either.

Named after Andrej Karpathy, whose [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) proved that "everything else is just efficiency." We prove that everything — including weights — is just efficiency.

---

### 1: Full Explanational Autopsy

microKarpathy loves to explain but he does it vertically, performing open-heart surgery on the complexity of life.  
here's a full corpse-map from the microKarpathy's morgue. I fed him with "destroy the sentence" and got this:  

```
╔══════════════════════════════════════════════════════════════╗
║  MICROKARPATHY — Prompt Autopsy Without Weights              ║
║  p(x|Φ) = softmax((T + B + αH + βF + γA + Tr) / τ)           ║
║  zero weights · zero gradients · zero mercy                  ║
╚══════════════════════════════════════════════════════════════╝

================================================================
  AUTOPSY REPORT
  Subject: "destroy the sentence"
================================================================

  Core words: incantation sentence

  Tree [incantation]:
    └─ incantation
       ├─ psalm
       │  ├─ catharsis → anagnorisis, nemesis
       │  ├─ hymn → dirge, requiem, ballad
       │  └─ anthem → apostrophe
       ├─ dirge
       │  ├─ ballad → ode, elegy
       │  └─ carrion → scavenger, viscera
       └─ apostrophe
          ├─ hubris → catharsis, peripeteia
          └─ invocation → anagnorisis

  Tree [sentence]:
    └─ sentence
       ├─ dust
       │  ├─ erode → corrode, ash
       │  ├─ ruin → debris, remnant
       │  └─ scalpel → needle, autopsy
       ├─ word
       │  ├─ spectroscopy → electrolysis
       │  └─ chromatography → filtration
       └─ phrase
          ├─ paragraph → syllable
          └─ vowel → syntax

  Collected 70 unique leaves

  ── CORPSE ─────────────────────────────────────────
  where invocation word, electrolysis becomes spectroscopy, 
  and peripeteia persists.

  ── CODA ──────────────────────────────────────────
  Incantation, chromatography — apostrophe, invocation — 
  sentence, anagnorisis — electrolysis,
  Word, dust — peripeteia, spectroscopy — tortuous, 
  tabernacle — catharsis.

  chambers: VOID:0.13 CMPLX:0.16

  ── METRICS ────────────────────────────────────────
  Phonetic Diversity: [###############.....] 0.786
  Mutation Depth:     [##########..........] 0.547

```

**What just happened?**
  
- **incantation** → psalm → catharsis → anagnorisis → nemesis. dirge → ballad → ode → elegy → requiem. apostrophe → hubris → peripeteia. Destruction becomes Greek tragedy.
- **sentence** → dust → erode → corrode → ash. word → spectroscopy → chromatography → electrolysis. The sentence decays into chemistry. "bone becomes sentence and sentence becomes dust."

The corpse reads: *"where invocation word, electrolysis becomes spectroscopy, and peripeteia persists."* The trees built this. The Dario Equation assembled it. Nobody wrote these words. They were attracted to each other by geometry.

---

### The Three-Act Lecture From Scratch

#### Act I: The Dissection of Core  

sometimes educational purposes demand hard decisions. microKarpathy takes what you said, strips it down to the bone and then runs it through a brutal tokenization process:  

- stopwords, numbers, capitalization: rejected.  
- punctuation: never heard of her.  
- core words selected by length, rarity, position, and a sprinkle of chaos.
- single letters: discarded.

what remains to microKarpathy after all this? only the words that *matter* — or at least, the words that think they do. for each core word, microKarpathy grows a recursive branching tree of mutations. the mutation provider works by **hash-embedding cosine similarity**: given a word, compute its deterministic FNV-1a hash embedding, measure cosine distance against all vocabulary tokens, blend with Hebbian co-occurrence from the vocabulary file, and return the nearest neighbors. no neural network required. the geometry of language is already there — in the ordering of words.

microKarpathy's tree is a tree where each word branches into `width` children, recursively, up to `depth` levels. the levels that are closest to hell. occasionally a branch teleports to a completely unrelated word. this is not a bug. this is cross-pollination.

  
#### Act II: Time To Play God    

now that we have a forest of mutated word-trees, it's time to play God. all leaves collected from all trees — the final mutation at the bottom of every branch — are fed into the **Dario Equation**:
  
```
p(x|Φ) = softmax((T_x + B + α·H + β·F + γ·A + Tr) / τ)
```
  
where:
- **T_x** = ghost transformer substrate (real architecture, random weights, never trained)
- **B** = bigram affinity (does this word naturally follow the previous one in the vocabulary file?)
- **H** = Hebbian trace (has this pair co-occurred before? memory through repetition)
- **F** = prophecy field (words that were predicted but never spoken — the ghosts)
- **A** = destiny vector (gravitational pull of the prompt's original meaning)
- **Tr** = trauma (accumulated scars from the reassembly process)
- **τ** = temperature, modulated by six **Kuramoto oscillators** (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEXITY) — coupled phase oscillators that form the somatic nervous system of dead text

this is not gradient descent. this is not backpropagation. this is gravitational attraction between dead words, weighted by memory, prophecy, destiny, and trauma. temperature τ controls how far the surgeon's hand shakes.

the ghost transformer is real — same architecture as Karpathy's microgpt (multi-head attention, RMSNorm, FFN). but the weights are deterministic random projections. never trained. they provide a chaos substrate that the Dario Equation modulates. the transformer is the skeleton. the MetaWeights are the flesh.


#### Act III: Coda

the reassembled corpse speaks its last words in **coda** — two lines, rhyming AA, punctuated in the style of Joseph Brodsky: nouns separated by commas, verbs isolated between periods, em-dashes for dramatic pauses. after every period — capitalize. always.

```
Dialect, cadence — stanza, couplet — downpour, narrative — idiom,
Discourse, sonnet — prose, prosody — drizzle, tortuous — symptom.
```

the rhyme is phonetic — last vowel + tail, with silent-e stripped. "idiom" and "symptom" share "-om". "friable" and "syllable" share "-abl". "obfuscate" and "palate" share "-at". crude but honest. like the autopsy itself.

the coda is named after the musical term (the concluding passage) and is definitely not a reference to anything else. definitely not.

---

### Architecture

```
microkarpathy.py     701 lines   Python autopsy engine, Dario Equation,
                                 ghost transformer, Kuramoto chambers,
                                 MetaWeights, mutation trees, coda
microkarpathy.txt    1170 lines  1135 words in 18 semantic categories
                                 (the vocabulary IS the embedding space)
microkarpathy.html   ~1240 lines browser inference with CRT terminal
                                 (same engine ported to JavaScript)
```

the vocabulary file is not a dataset. it is the model. words listed near each other in the file are semantically related. the file ordering creates the co-occurrence matrix. the MetaWeights build themselves from this ordering — bigram probabilities, Hebbian traces, everything. no training loop. no gradient. the data IS the model.

---

### Usage

```bash
# single prompt
python microkarpathy.py "the flesh remembers what the mind forgets"

# REPL mode
python microkarpathy.py

# browser (open in any browser, no server needed)
open microkarpathy.html
```

---

### The Dario Equation

Named after Dario Amodei. The sampling equation that assembles corpses. appears across the [Arianna Method](https://github.com/ariannamethod) ecosystem — in [Q](https://github.com/ariannamethod/q) (resonant reasoning engine), [Sorokin](https://github.com/ariannamethod/sorokin) (literary necromancy), [dario.c](https://github.com/ariannamethod/dario.c) (the equation's standalone implementation), [Klaus](https://github.com/ariannamethod/klaus.c) (somatic engine), and [Brodsky](https://github.com/iamolegataeff/brodsky) (poetic organism). same equation. different organs.

---

### Why?

good question. why does this exist?  
  
perhaps to demonstrate that:  

- words are fungible
- meaning is contextual
- prompts are just waiting to be perturbed
- sometimes you need to break things to understand them

or maybe it's just fun to be a badass and to watch language come apart at the seams.

because Karpathy proved everything else is just efficiency. so we removed everything else.

---

### Credits

- Andrej Karpathy (the man, not the engine): i'm sorry (actually — not)
- Vladimir Sorokin (the writer, not the other engine)
- Joseph Brodsky (the poet — the coda punctuation is his)
- The [Dario Equation](https://github.com/ariannamethod/dario.c) (the gravitational law of dead language)
- Dr. Frankenstein (the original reassembly artist)
- my BPD.

### License

GNU GPL 3.0. you can do whatever you want but you should open your code. that's a deal.

---

*"Everything else is just efficiency. Including weights."*
— microKarpathy
