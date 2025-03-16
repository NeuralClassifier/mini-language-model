# Self-Attention

### Linear Projections:
Given an input matrix X of shape [seq_length, embed_dim], we generate queries (Q), keys (K), and values (V) using learned weight matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where each weight matrix $W_Q$, $W_K$, $W_V$ has shape [embed_dim, head_dim]

### Scaled Dot-Product Attention:
Compute the attention scores:

$$
scores = \frac{QK^T}{\sqrt{d_k}}
$$

where $d_k$ is the dimension of the key vectors (**embed_dim** in our case).

### Softmax:
Normalize the scores using the softmax function:

$$
attention_{weights} = softmax(\text{scores})
$$

### Weighted Sum:
Compute the final output by applying the attention weights to **V**:

$$
output = attention_{weights} \times V
$$

This process allows the model to focus on different parts of the input sequence dynamically, capturing contextual information effectively.


In this step we compute the attention scores of every word in the text to have its contextual representation. Self-attention is a tool to determine how much focus or importance one word (or token) should give to another word (or token) in the sentence when building its contextual representation. It’s almost like asking, "Which parts of the sentence should I pay more attention to when understanding the meaning of this word?"

The attention mechanism works through three key components:

Query (Q): The word that is asking for attention. It's the token whose representation is being built.
Key (K): The words that are considered as possible sources of information for the query.
Value (V): The actual information from the words (keys) that will be passed to the query.

The attention score is essentially how much the query (a word in the sentence or block of text) should pay attention to each key (the other words in the same sentence or a text). This score is computed by taking the similarity between the query and the key. For each word in the sentence/text, we calculate attention scores by comparing the query (Q) with all the keys (K):

$$
Attention_{score} = \frac{Q.K^T}{\sqrt{d_k}}
$$

$Q$ is the query vector (the word you're trying to represent), $K^T$ is the transpose of the key vector (the other words that are candidates for providing context), and $d_k$ is a scaling factor, usually the dimension of the key vectors, to avoid large numbers that could destabilize learning. **The dot product $Q.K^T$** measures how similar the query is to each key. Higher similarity means that the word the query is referring to has more relevance to it. **The division by $\sqrt{d_k}$** ensures that the dot product values don’t get too large, helping to keep the attention scores in a manageable range. This is just a normalization step. After calculating the attention scores, **we apply a softmax function** to convert them into probabilities. This step ensures that all attention scores add up to 1 (like a probability distribution), making them interpretable as how much attention each word should give to every other word.


# Attention Mechanism Example

This example walks through a simple case of computing attention scores in a transformer model.

## Example Sentence

Consider the sentence:

```
Alice met Bob at the cafe.
```

We want to compute the attention scores for the word **"met"** (the query) based on the rest of the sentence (the keys).

### Defining Query and Keys
- **Query (Q):** "met"
- **Keys (K):** "Alice", "met", "Bob", "at", "the", "cafe"

### Vector Representations
Lets just assume that every word is made as a simple vector after being processed by the language model:

| Word  | Vector (Q or K)     |
|--------|------------------|
| Alice  | [0.9, 0.1, 0.2] |
| met    | [0.1, 0.8, 0.3] |
| Bob    | [0.7, 0.5, 0.1] |
| at     | [0.4, 0.2, 0.6] |
| the    | [0.2, 0.3, 0.8] |
| cafe   | [0.3, 0.7, 0.9] |

## Computing Attention Scores
The attention score is computed as the dot product between **Q (met)** and each **K (word in the sentence)**.

### Dot Product Calculations

#### 1. "met" and "Alice"
```
0.1 × 0.9 + 0.8 × 0.1 + 0.3 × 0.2 = 0.23
```

#### 2. "met" and "met"
```
0.1 × 0.1 + 0.8 × 0.8 + 0.3 × 0.3 = 0.74
```

#### 3. "met" and "Bob"
```
0.1 × 0.7 + 0.8 × 0.5 + 0.3 × 0.1 = 0.53
```

#### 4. "met" and "at"
```
0.1 × 0.4 + 0.8 × 0.2 + 0.3 × 0.6 = 0.38
```

#### 5. "met" and "the"
```
0.1 × 0.2 + 0.8 × 0.3 + 0.3 × 0.8 = 0.47
```

#### 6. "met" and "cafe"
```
0.1 × 0.3 + 0.8 × 0.7 + 0.3 × 0.9 = 0.86
```

## Summary of Attention Scores
| Word  | Attention Score |
|--------|----------------|
| Alice  | 0.23          |
| met    | 0.74          |
| Bob    | 0.53          |
| at     | 0.38          |
| the    | 0.47          |
| cafe   | 0.86          |

The highest attention score is assigned to **"cafe"**, meaning that in this context, "met" is most strongly related to "cafe" based on the given vector representations.

## Step 2: Applying Softmax
Now, we apply the softmax function to the dot products to get the final attention scores. The softmax function takes the raw scores (dot products) and turns them into probabilities.

### Softmax Calculation
#### Given raw scores:
```
[0.23, 0.74, 0.53, 0.38, 0.47, 0.86]
```

#### Exponentiate each raw score:
```
exp(0.23) = 1.258
exp(0.74) = 2.101
exp(0.53) = 1.698
exp(0.38) = 1.462
exp(0.47) = 1.599
exp(0.86) = 2.365
```

#### Sum the exponentiated values:
```
1.258 + 2.101 + 1.698 + 1.462 + 1.599 + 2.365 = 10.483
```

#### Compute softmax probabilities:
```
Softmax([0.23, 0.74, 0.53, 0.38, 0.47, 0.86]) = [
 1.258 / 10.483,
 2.101 / 10.483,
 1.698 / 10.483,
 1.462 / 10.483,
 1.599 / 10.483,
 2.365 / 10.483
]
```

#### Resulting attention scores:
```
[0.12, 0.20, 0.16, 0.14, 0.15, 0.23]
```
