# mini-language-model


## Embedding

Tokens created during the vocabbulary mapping step are categorical variables with no inherent numerical relationship. Deep learning models work with real-valued vectors, and not discreet indices. 

The index values $t_i$ are arbitrary (e.g., "hello" could be index 0, "world" index 1, etc.). There is no meaningful numerical relationship between these indices (e.g. $t_1 = 0$, ​and $t_2 = 1$ do not imply similarity). These discreet labels does not have a natural order or similarity measure. Computing differentiability through gradients cannot be done using categorical indices. One hot encoding is one option, however, one-hot vectors treat every token as equally distant from every other token. There’s no notion of similarity built into the representation. Thus, we need a mapping from discrete token indices to a continuous vector space. We therefore learn an embedding function $h: Z \to R^d$ which maps each token index $x_i$ to a dense, continuous representation $h(x_i)$:

$$
h(x_i) = E[x_i]
$$

and we create:

$$
H = 
\begin{bmatrix}
v_1 \\
v_2 \\
v_3 \\
\vdots \\
v_n
\end{bmatrix}
$$
