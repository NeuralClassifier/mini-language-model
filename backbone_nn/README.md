# Embedding

Once the Vocabulary Map is created, each token $t_i$ is mapped to an embedding matrix $E$:

$$
E = T^x \to R^d
$$

where, $d$ is the embedding dimension, and embedding $E(t_i)$ for each token $t_i$ is:

$$
h_i = E(t_i) \in R^d
$$

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
