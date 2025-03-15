# mini-language-model


## Embedding

Tokens created during the vocabbulary mapping step are categorical variables with no inherent numerical relationship. Deep learning models work with real-valued vectors, and not discreet indices. 

The index values $t_i$ are arbitrary (e.g., "hello" could be index 0, "world" index 1, etc.). There is no meaningful numerical relationship between these indices (e.g. $t_1 = 0$, ​and $t_2 = 1$ do not imply similarity). Computing differentiability through gradients cannot be done using categorical indices. Thus, we need a mapping from discrete token indices to a continuous vector space. We therefore learn an embedding function $h: Z \to R^d$ which maps each token index $x_i$ to a dense, continuous representation $h(x_i)$:

$$
h(x_i) = E[x_i]
$$
