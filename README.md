# mini-language-model


## Embedding

Tokens created during the vocabbulary mapping step are categorical variables with no inherent numerical relationship. Deep learning models work with real-valued vectors, and not discreet indices. 

The index values $t_i$ are arbitrary (e.g., "hello" could be index 0, "world" index 1, etc.). There is no meaningful numerical relationship between these indices (e.g. $t_1 = 0$, ​and $t_2 = 1$ do not imply similarity). These discreet labels does not have a natural order or similarity measure. Also, computing differentiability through gradients cannot be done using categorical indices. One hot encoding is one option, however, one-hot vectors treat every token as equally distant from every other token. There’s no notion of similarity built into the representation. Thus, we need a mapping from discrete token indices to a continuous vector space. We therefore learn an embedding function $h: Z \to R^d$ which maps each token index $x_i$ to a dense, continuous representation $h(x_i)$:

$$
h(x_i) = E[x_i]
$$

and we create:

$$
H = 
\begin{bmatrix}
h(x_1) \\
h(x_2) \\
h(x_3) \\
\vdots \\
h(x_4)
\end{bmatrix}
$$

The embeddings that are learnt here after the traing are contextualized.

## Self-Attention

In this step we compute the attention scores of every word in the text to have its contextual representation. Self-attention is a tool to determine how much focus or importance one word (or token) should give to another word (or token) in the sentence when building its contextual representation. It’s almost like asking, "Which parts of the sentence should I pay more attention to when understanding the meaning of this word?"

The attention mechanism works through three key components:

Query (Q): The word that is asking for attention. It's the token whose representation is being built.
Key (K): The words that are considered as possible sources of information for the query.
Value (V): The actual information from the words (keys) that will be passed to the query.

The attention score is essentially how much the query (a word in the sentence or block of text) should pay attention to each key (the other words in the same sentence or a text). This score is computed by taking the similarity between the query and the key. For each word in the sentence/text, we calculate attention scores by comparing the query (Q) with all the keys (K):

$$
Attention_{score} = \frac{Q.K^T}{\sqrt{d_k}}
$$

$Q$ is the query vector (the word you're trying to represent), $K^T$ is the transpose of the key vector (the other words that are candidates for providing context), and $d_k$ is a scaling factor, usually the dimension of the key vectors, to avoid large numbers that could destabilize learning.
