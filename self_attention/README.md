# Self-Attention

### Linear Projections:
Given an input matrix **X** of shape **[seq_length, embed_dim]**, we generate **queries (Q)**, **keys (K)**, and **values (V)** using learned weight matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where each weight matrix $W_Q$, $W_K$, $W_V$ has shape [embed_dim, head_dim]

### Scaled Dot-Product Attention:
Compute the attention scores:

$$
scores = \frac{QK^T}{\sqrt{d_k}}
$$

where **d_k** is the dimension of the key vectors (**embed_dim** in our case).

### Softmax:
Normalize the scores using the softmax function:

$$
attention_weights = \text{softmax}(\text{scores})
$$

### Weighted Sum:
Compute the final output by applying the attention weights to **V**:

$$
output = attention_weights \times V
$$

This process allows the model to focus on different parts of the input sequence dynamically, capturing contextual information effectively.
