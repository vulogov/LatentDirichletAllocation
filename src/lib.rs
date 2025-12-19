extern crate log;

use rand::Rng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};

fn default_stopwords() -> HashSet<&'static str> {
    [
        "a","an","and","are","as","at","be","by","for","from","has","in",
        "is","it","of","on","or","that","the","to","was","were","will","with",
        "http","https","ftp","s3"
    ]
    .into_iter()
    .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    let stop = default_stopwords();
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphabetic() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|tok| tok.len() >= 2 && !stop.contains(*tok))
        .map(|tok| tok.to_string())
        .collect()
}

type WordId = usize;
type TopDocuments = HashMap<usize, HashMap<String, f64>>;

pub fn default(docs_raw: &[&str]) -> Lda {
    Lda::from_documents(3, 0.1, 0.01, docs_raw, 42)
}

pub struct Lda {
    // hyper-parameters
    pub k:      usize,      // number of topics
    pub alpha:  f64,        // Dirichlet prior for document-topic distributions
    pub beta:   f64,        // Dirichlet prior for topic-word distributions

    // corpus and vocabulary
    vocab: Vec<String>,
    pub word_to_id: HashMap<String, WordId>,
    docs: Vec<Vec<WordId>>, // tokenized docs as word IDs

    // latent variables & counts
    z: Vec<Vec<usize>>,          // topic assignment for each word position
    ndk: Vec<Vec<usize>>,        // [doc][topic]: # tokens in doc assigned to topic
    nkw: Vec<Vec<usize>>,        // [topic][word]: # occurrences of word in topic
    nk: Vec<usize>,              // [topic]: total tokens assigned to topic

    // cached for normalization
    doc_lengths: Vec<usize>,     // # tokens in each doc

    // rng
    rng: StdRng,
}

impl Lda {
    /// Create an LDA model from raw text documents.
    pub fn from_documents(k: usize, alpha: f64, beta: f64, docs_raw: &[&str], seed: u64) -> Self {
        // 1) tokenize documents
        let tokenized: Vec<Vec<String>> = docs_raw.iter().map(|d| tokenize(d)).collect();

        // 2) build vocabulary
        let mut word_to_id: HashMap<String, WordId> = HashMap::new();
        let mut vocab: Vec<String> = Vec::new();
        for doc in &tokenized {
            for w in doc {
                if !word_to_id.contains_key(w) {
                    let id = vocab.len();
                    vocab.push(w.clone());
                    word_to_id.insert(w.clone(), id);
                }
            }
        }

        // 3) convert docs to word IDs
        let docs: Vec<Vec<WordId>> = tokenized
            .into_iter()
            .map(|doc| doc.into_iter().map(|w| word_to_id[&w]).collect())
            .collect();

        // 4) allocate counts
        let v = vocab.len();
        let d = docs.len();
        let mut ndk = vec![vec![0usize; k]; d];
        let mut nkw = vec![vec![0usize; v]; k];
        let mut nk = vec![0usize; k];
        let mut z = vec![vec![0usize; 0]; d];
        let doc_lengths = docs.iter().map(|x| x.len()).collect::<Vec<_>>();

        let mut rng = StdRng::seed_from_u64(seed);

        // 5) random initialization of topic assignments
        for (di, doc) in docs.iter().enumerate() {
            z[di] = vec![0; doc.len()];
            for (pi, &w) in doc.iter().enumerate() {
                let topic = rng.gen_range(0..k);
                z[di][pi] = topic;
                ndk[di][topic] += 1;
                nkw[topic][w] += 1;
                nk[topic] += 1;
            }
        }

        Self {
            k,
            alpha,
            beta,
            vocab,
            word_to_id,
            docs,
            z,
            ndk,
            nkw,
            nk,
            doc_lengths,
            rng,
        }
    }

    /// Perform collapsed Gibbs sampling for `iters` iterations.
    pub fn train(&mut self, iters: usize) {
        let v = self.vocab.len();
        let vb = (v as f64) * self.beta;

        for it in 0..iters {
            for di in 0..self.docs.len() {
                for pi in 0..self.docs[di].len() {
                    let w = self.docs[di][pi];
                    let old_t = self.z[di][pi];

                    // Decrement old counts
                    self.ndk[di][old_t] -= 1;
                    self.nkw[old_t][w] -= 1;
                    self.nk[old_t] -= 1;

                    // Compute conditional probabilities for topics
                    // p(t) ∝ (ndk[d][t] + alpha) * (nkw[t][w] + beta) / (nk[t] + V*beta)
                    let mut weights = vec![0.0f64; self.k];
                    for t in 0..self.k {
                        let left = (self.ndk[di][t] as f64) + self.alpha;
                        let right_num = (self.nkw[t][w] as f64) + self.beta;
                        let right_den = (self.nk[t] as f64) + vb;
                        weights[t] = left * (right_num / right_den);
                    }

                    // Sample new topic
                    // Normalize via WeightedIndex (works with non-negative weights)
                    let sumw: f64 = weights.iter().sum();
                    // fall back if all weights are zero (unlikely but safe)
                    let new_t = if sumw <= f64::EPSILON {
                        self.rng.gen_range(0..self.k)
                    } else {
                        let wi = WeightedIndex::new(&weights).unwrap();
                        wi.sample(&mut self.rng)
                    };

                    // Assign and increment counts
                    self.z[di][pi] = new_t;
                    self.ndk[di][new_t] += 1;
                    self.nkw[new_t][w] += 1;
                    self.nk[new_t] += 1;
                }
            }

            if (it + 1) % 50 == 0 {
                log::debug!("Trainig LDA: iteration {}/{}", it + 1, iters);
            }
        }
    }

    /// θ[d][t] = (ndk[d][t] + α) / (N_d + K*α)
    pub fn theta(&self) -> Vec<Vec<f64>> {
        let mut theta = vec![vec![0.0f64; self.k]; self.docs.len()];
        for d in 0..self.docs.len() {
            let denom = (self.doc_lengths[d] as f64) + (self.k as f64) * self.alpha;
            for t in 0..self.k {
                theta[d][t] = ((self.ndk[d][t] as f64) + self.alpha) / denom;
            }
        }
        theta
    }

    /// φ[t][w] = (nkw[t][w] + β) / (nk[t] + V*β)
    pub fn phi(&self) -> Vec<Vec<f64>> {
        let v = self.vocab.len();
        let vb = (v as f64) * self.beta;

        let mut phi = vec![vec![0.0f64; v]; self.k];
        for t in 0..self.k {
            let denom = (self.nk[t] as f64) + vb;
            for w in 0..v {
                phi[t][w] = ((self.nkw[t][w] as f64) + self.beta) / denom;
            }
        }
        phi
    }

    /// Return top `n` words for each topic by φ[t][w].
    pub fn top_words(&self, n: usize) -> Vec<Vec<(String, f64)>> {
        let phi = self.phi();
        let mut tops = Vec::with_capacity(self.k);
        for t in 0..self.k {
            let mut pairs: Vec<(usize, f64)> = (0..self.vocab.len())
                .map(|w| (w, phi[t][w]))
                .collect();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let best = pairs
                .into_iter()
                .take(n)
                .map(|(w, p)| (self.vocab[w].clone(), p))
                .collect::<Vec<_>>();
            tops.push(best);
        }
        tops
    }

    /// Convert "top wors" into HashMap
    pub fn topics(&self, topn: usize) -> TopDocuments {
        let mut res: HashMap<usize, HashMap<String, f64>> = HashMap::new();
        for (t, words) in self.top_words(topn).into_iter().enumerate() {
            let mut row: HashMap<String, f64> = HashMap::new();
            for (word, score) in words {
                row.insert(word.to_string(), score);
            }
            res.insert(t, row);
        }
        return res;
    }
}
