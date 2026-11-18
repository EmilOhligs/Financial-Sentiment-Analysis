/*
 * models.js — the two classifiers from this repository, in the browser.
 *
 * vader():  line-by-line port of nltk.sentiment.vader (NLTK 3.x), including
 *           its quirks, so scores match the Python version used in the study.
 * logreg(): TF-IDF + logistic regression scored from model.json, mirroring
 *           scikit-learn's TfidfVectorizer and LogisticRegression.predict_proba.
 *
 * MIT License. VADER: Hutto & Gilbert (2014), MIT License.
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- VADER
  const B_INCR = 0.293, B_DECR = -0.293, C_INCR = 0.733, N_SCALAR = -0.74;

  const NEGATE = new Set(("aint arent cannot cant couldnt darent didnt doesnt ain't aren't can't " +
    "couldn't daren't didn't doesn't dont hadnt hasnt havent isnt mightnt mustnt neither don't " +
    "hadn't hasn't haven't isn't mightn't mustn't neednt needn't never none nope nor not nothing " +
    "nowhere oughtnt shant shouldnt uhuh wasnt werent oughtn't shan't shouldn't uh-uh wasn't " +
    "weren't without wont wouldnt won't wouldn't rarely seldom despite").split(" "));

  const BOOSTER = {};
  ("absolutely amazingly awfully completely considerably decidedly deeply effing enormously " +
   "entirely especially exceptionally extremely fabulously flipping flippin fricking frickin " +
   "frigging friggin fully fucking greatly hella highly hugely incredibly intensely majorly more " +
   "most particularly purely quite really remarkably so substantially thoroughly totally " +
   "tremendously uber unbelievably unusually utterly very").split(" ").forEach(w => BOOSTER[w] = B_INCR);
  ["almost", "barely", "hardly", "just enough", "kind of", "kinda", "kindof", "kind-of", "less",
   "little", "marginally", "occasionally", "partly", "scarcely", "slightly", "somewhat", "sort of",
   "sorta", "sortof", "sort-of"].forEach(w => BOOSTER[w] = B_DECR);
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

  const IDIOMS = {
    "the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2,
    "cut the mustard": 2, "kiss of death": -1.5, "hand to mouth": -2,
  };

  const PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";  // Python string.punctuation
  const PUNCT_RE = new RegExp("[" + PUNCT.replace(/[\\\]\-^]/g, "\\$&") + "]", "g");
  const PUNC_LIST = [".", "!", "?", ",", ";", ":", "-", "'", "\"", "!!", "!!!", "??", "???",
    "?!?", "!?!", "?!?!", "!?!?"];

  // Python str.isupper(): at least one cased character and no lowercase ones.
  function isUpper(s) {
    return s !== s.toLowerCase() && s === s.toUpperCase();
  }
  // Python str.split() without arguments.
  const pySplit = (s) => s.split(/\s+/).filter(Boolean);

  function negated(word) {
    const w = word.toLowerCase();
    return NEGATE.has(w) || w.includes("n't");
  }

  function scalarIncDec(word, valence, isCapDiff) {
    const wl = word.toLowerCase();
    let scalar = 0;
    if (has(BOOSTER, wl)) {
      scalar = BOOSTER[wl];
      if (valence < 0) scalar *= -1;
      if (isUpper(word) && isCapDiff) scalar += valence > 0 ? C_INCR : -C_INCR;
    }
    return scalar;
  }

  function tokenize(text) {
    const wordsOnly = new Set(pySplit(text.replace(PUNCT_RE, "")).filter(w => w.length > 1));
    const map = {};
    for (const p of PUNC_LIST) for (const w of wordsOnly) map[p + w] = w;   // punctuation before
    for (const w of wordsOnly) for (const p of PUNC_LIST) map[w + p] = w;   // punctuation after
    return pySplit(text).filter(t => t.length > 1).map(t => (has(map, t) ? map[t] : t));
  }

  function vader(text, lexicon) {
    const words = tokenize(text);
    const caps = words.filter(isUpper).length;
    const isCapDiff = words.length - caps > 0 && words.length - caps < words.length;
    const lex = (w) => has(lexicon, w);
    const first = {};
    words.forEach((t, i) => { if (!has(first, t)) first[t] = i; });

    function neverCheck(valence, startI, i) {
      if (startI === 0) {
        if (negated(words[i - 1])) valence *= N_SCALAR;
      } else if (startI === 1) {
        if (words[i - 2] === "never" && (words[i - 1] === "so" || words[i - 1] === "this")) valence *= 1.5;
        else if (negated(words[i - 2])) valence *= N_SCALAR;
      } else if (startI === 2) {
        if ((words[i - 3] === "never" && (words[i - 2] === "so" || words[i - 2] === "this")) ||
            (words[i - 1] === "so" || words[i - 1] === "this")) valence *= 1.25;
        else if (negated(words[i - 3])) valence *= N_SCALAR;
      }
      return valence;
    }

    // Python negative indexing, used by the idiom check for i < 3.
    const at = (k) => words[k < 0 ? words.length + k : k];

    function idiomsCheck(valence, i) {
      const onezero = `${at(i - 1)} ${at(i)}`;
      const twoonezero = `${at(i - 2)} ${at(i - 1)} ${at(i)}`;
      const twoone = `${at(i - 2)} ${at(i - 1)}`;
      const threetwoone = `${at(i - 3)} ${at(i - 2)} ${at(i - 1)}`;
      const threetwo = `${at(i - 3)} ${at(i - 2)}`;
      for (const seq of [onezero, twoonezero, twoone, threetwoone, threetwo]) {
        if (has(IDIOMS, seq)) { valence = IDIOMS[seq]; break; }
      }
      if (words.length - 1 > i) {
        const zeroone = `${words[i]} ${words[i + 1]}`;
        if (has(IDIOMS, zeroone)) valence = IDIOMS[zeroone];
      }
      if (words.length - 1 > i + 1) {
        const zeroonetwo = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
        if (has(IDIOMS, zeroonetwo)) valence = IDIOMS[zeroonetwo];
      }
      if (has(BOOSTER, threetwo) || has(BOOSTER, twoone)) valence += B_DECR;
      return valence;
    }

    function leastCheck(valence, i) {
      if (i > 1 && !lex(words[i - 1].toLowerCase()) && words[i - 1].toLowerCase() === "least") {
        const w2 = words[i - 2].toLowerCase();
        if (w2 !== "at" && w2 !== "very") valence *= N_SCALAR;
      } else if (i > 0 && !lex(words[i - 1].toLowerCase()) && words[i - 1].toLowerCase() === "least") {
        valence *= N_SCALAR;
      }
      return valence;
    }

    const sentiments = [];
    for (const item of words) {
      const i = first[item];  // NLTK uses the first occurrence of a repeated token
      const il = item.toLowerCase();
      if ((i < words.length - 1 && il === "kind" && words[i + 1].toLowerCase() === "of") || has(BOOSTER, il)) {
        sentiments.push(0);
        continue;
      }
      let valence = 0;
      if (lex(il)) {
        valence = lexicon[il];
        if (isUpper(item) && isCapDiff) valence += valence > 0 ? C_INCR : -C_INCR;
        for (let s = 0; s < 3; s++) {
          if (i > s && !lex(words[i - (s + 1)].toLowerCase())) {
            let sc = scalarIncDec(words[i - (s + 1)], valence, isCapDiff);
            if (s === 1 && sc !== 0) sc *= 0.95;
            if (s === 2 && sc !== 0) sc *= 0.9;
            valence += sc;
            valence = neverCheck(valence, s, i);
            if (s === 2) valence = idiomsCheck(valence, i);
          }
        }
        valence = leastCheck(valence, i);
      }
      sentiments.push(valence);
    }

    // "but" check
    const lower = words.map(w => w.toLowerCase());
    const bi = lower.indexOf("but");
    if (bi >= 0) {
      for (let k = 0; k < sentiments.length; k++) {
        if (k < bi) sentiments[k] *= 0.5;
        else if (k > bi) sentiments[k] *= 1.5;
      }
    }

    let compound = 0;
    if (sentiments.length) {
      let sum = sentiments.reduce((a, b) => a + b, 0);
      const ep = Math.min((text.match(/!/g) || []).length, 4) * 0.292;
      const qmCount = (text.match(/\?/g) || []).length;
      const qm = qmCount > 1 ? (qmCount <= 3 ? qmCount * 0.18 : 0.96) : 0;
      if (sum > 0) sum += ep + qm;
      else if (sum < 0) sum -= ep + qm;
      compound = sum / Math.sqrt(sum * sum + 15);
    }
    return {
      compound: Math.round(compound * 1e4) / 1e4,
      tokens: words.map((w, k) => ({ word: w, valence: sentiments[k] })),
    };
  }

  function vaderLabel(compound, model) {
    if (compound >= model.vader.pos_threshold) return "positive";
    if (compound <= model.vader.neg_threshold) return "negative";
    return "neutral";
  }

  // --------------------------------------------------- TF-IDF + LogReg
  const TOKEN_RE = /[\p{L}\p{N}_]+/gu;  // runs of \w; scikit-learn keeps those of length >= 2

  function stripAccents(s) {
    return s.normalize("NFKD").replace(/\p{M}/gu, "");
  }

  function logreg(text, model) {
    const doc = stripAccents(text.toLowerCase());
    const tokens = (doc.match(TOKEN_RE) || []).filter(t => [...t].length >= 2);
    const grams = tokens.slice();
    if (model.ngram_range[1] >= 2) for (let k = 0; k + 1 < tokens.length; k++) grams.push(tokens[k] + " " + tokens[k + 1]);

    const counts = new Map();
    for (const g of grams) if (has(model.vocab, g)) counts.set(g, (counts.get(g) || 0) + 1);
    const vec = new Map();
    for (const [g, c] of counts) {
      const tf = model.sublinear_tf ? 1 + Math.log(c) : c;
      vec.set(g, tf * model.vocab[g].idf);
    }
    let norm = 0;
    for (const v of vec.values()) norm += v * v;
    norm = Math.sqrt(norm) || 1;

    const logits = model.intercept.slice();
    const contrib = [];
    for (const [g, v] of vec) {
      const x = v / norm;
      const w = model.vocab[g].coef;
      for (let k = 0; k < 3; k++) logits[k] += x * w[k];
      contrib.push({ term: g, x, coef: w });
    }
    const mx = Math.max(...logits);
    const ex = logits.map(z => Math.exp(z - mx));
    const tot = ex.reduce((a, b) => a + b, 0);
    const proba = ex.map(e => e / tot);
    const best = proba.indexOf(Math.max(...proba));
    // A term's push toward the predicted class, relative to the class average.
    const terms = contrib.map(c => ({
      term: c.term,
      push: c.x * (c.coef[best] - (c.coef[0] + c.coef[1] + c.coef[2]) / 3),
    })).sort((a, b) => b.push - a.push);
    return { label: model.labels[best], proba, terms, known: vec.size };
  }

  const api = { vader, vaderLabel, logreg, tokenize };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FSA = api;
})(typeof window !== "undefined" ? window : this);
