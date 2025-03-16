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
