from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def TED_generator(vocab_size):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


    def positional_encoding(position, d_model):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


    # ## Masking

    # Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. The mask indicates where pad value `0` is present: it outputs a `1` at those locations, and a `0` otherwise.

    # In[ ]:


    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



    # The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.
    #
    # This means that to predict the third word, only the first and second word will be used. Similarly to predict the fourth word, only the first, second and the third word will be used and so on.

    # In[ ]:


    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)



    # ## Scaled dot product attention

    # <img src="https://www.tensorflow.org/images/tutorials/transformer/scaled_attention.png" width="500" alt="scaled_dot_product_attention">
    #
    # The attention function used by the transformer takes three inputs: Q (query), K (key), V (value). The equation used to calculate the attention weights is:
    #
    # $$\Large{Attention(Q, K, V) = softmax_k(\frac{QK^T}{\sqrt{d_k}}) V} $$
    #
    # The dot-product attention is scaled by a factor of square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients resulting in a very hard softmax.
    #
    # For example, consider that `Q` and `K` have a mean of 0 and variance of 1. Their matrix multiplication will have a mean of 0 and variance of `dk`. Hence, *square root of `dk`* is used for scaling (and not any other number) because the matmul of `Q` and `K` should have a mean of 0 and variance of 1, so that we get a gentler softmax.
    #
    # The mask is multiplied with *-1e9 (close to negative infinity).* This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately before a softmax. The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.

    # In[ ]:


    def scaled_dot_product_attention(q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth) (batch_size, num_heads, seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth) (batch_size, num_heads, seq_len_q, depth)
          v: value shape == (..., seq_len_v, depth_v) (batch_size, num_heads, seq_len_q, depth)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

        return output, attention_weights


    # As the softmax normalization is done on K, its values decide the amount of importance given to Q.
    #
    # The output represents the multiplication of the attention weights and the V (value) vector. This ensures that the words we want to focus on are kept as is and the irrelevant words are flushed out.

    # In[ ]:

    # ## Multi-head attention

    # <img src="https://www.tensorflow.org/images/tutorials/transformer/multi_head_attention.png" width="500" alt="multi-head attention">
    #
    #
    # Multi-head attention consists of four parts:
    # *    Linear layers and split into heads.
    # *    Scaled dot-product attention.
    # *    Concatenation of heads.
    # *    Final linear layer.

    # Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear (Dense) layers and split up into multiple heads.
    #
    # The `scaled_dot_product_attention` defined above is applied to each head (broadcasted for efficiency). An appropriate mask must be used in the attention step.  The attention output for each head is then concatenated (using `tf.transpose`, and `tf.reshape`) and put through a final `Dense` layer.
    #
    # Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to jointly attend to information at different positions from different representational spaces. After the split each head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full dimensionality.

    # In[ ]:


    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model % self.num_heads == 0

            self.depth = d_model // self.num_heads

            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

            self.dense = tf.keras.layers.Dense(d_model)

        def split_heads(self, x, batch_size):
            """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, v, k, q, mask):
            batch_size = tf.shape(q)[0]

            q = self.wq(q)  # (batch_size, seq_len, d_model)
            k = self.wk(k)  # (batch_size, seq_len, d_model)
            v = self.wv(v)  # (batch_size, seq_len, d_model)

            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

            # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
            # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)

            scaled_attention = tf.transpose(scaled_attention,
                                            perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

            concat_attention = tf.reshape(scaled_attention,
                                          (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

            output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

            return output, attention_weights


    # Create a `MultiHeadAttention` layer to try out. At each location in the sequence, `y`, the `MultiHeadAttention` runs all 8 attention heads across all other locations in the sequence, returning a new vector of the same length at each location.

    # In[


    # ## Point wise feed forward network

    # Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.

    # In[ ]:


    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])



    # ## Encoder and decoder

    # <img src="https://www.tensorflow.org/images/tutorials/transformer/transformer.png" width="600" alt="transformer">

    # The transformer model follows the same general pattern as a standard [sequence to sequence with attention model](nmt_with_attention.ipynb).
    #
    # * The input sentence is passed through `N` encoder layers that generates an output for each word/token in the sequence.
    # * The decoder attends on the encoder's output and its own input (self-attention) to predict the next word.

    # ### Encoder layer
    #
    # Each encoder layer consists of sublayers:
    #
    # 1.   Multi-head attention (with padding mask)
    # 2.    Point wise feed forward networks.
    #
    # Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections help in avoiding the vanishing gradient problem in deep networks.
    #
    # The output of each sublayer is `LayerNorm(x + Sublayer(x))`. The normalization is done on the `d_model` (last) axis. There are N encoder layers in the transformer.

    # In[ ]:


    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(EncoderLayer, self).__init__()

            self.mha = MultiHeadAttention(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)

            self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

            ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

            return out2


    # ### Decoder layer
    #
    # Each decoder layer consists of sublayers:
    #
    # 1.   Masked multi-head attention (with look ahead mask and padding mask)
    # 2.   Multi-head attention (with padding mask). V (value) and K (key) receive the *encoder output* as inputs. Q (query) receives the *output from the masked multi-head attention sublayer.*
    # 3.   Point wise feed forward networks
    #
    # Each of these sublayers has a residual connection around it followed by a layer normalization. The output of each sublayer is `LayerNorm(x + Sublayer(x))`. The normalization is done on the `d_model` (last) axis.
    #
    # There are N decoder layers in the transformer.
    #
    # As Q receives the output from decoder's first attention block, and K receives the encoder output, the attention weights represent the importance given to the decoder's input based on the encoder's output. In other words, the decoder predicts the next word by looking at the encoder output and self-attending to its own output. See the demonstration above in the scaled dot product attention section.

    # In[ ]:


    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(DecoderLayer, self).__init__()

            self.mha1 = MultiHeadAttention(d_model, num_heads)
            self.mha2 = MultiHeadAttention(d_model, num_heads)

            self.ffn = point_wise_feed_forward_network(d_model, dff)

            self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            self.layernorm3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.dropout3 = tf.keras.layers.Dropout(rate)

        def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
            # enc_output.shape == (batch_size, input_seq_len, d_model)
            # x = the output of previous decoder layer (initially it will just be the target sequence)

            attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)

            attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

            ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

            return out3, attn_weights_block1, attn_weights_block2


    # ### Encoder
    #
    # The `Encoder` consists of:
    # 1.   Input Embedding
    # 2.   Positional Encoding
    # 3.   N encoder layers
    #
    # The input is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the encoder layers. The output of the encoder is the input to the decoder.

    # In[ ]:


    class Encoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                     embedding, rate=0.1):
            super(Encoder, self).__init__()

            self.d_model = d_model
            self.num_layers = num_layers

            self.embedding = embedding
            self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

            # dff is basically the number of units in the intermediate dense layer
            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]

            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            seq_len = tf.shape(x)[1]

            # adding embedding and position encoding.
            x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]

            x = self.dropout(x, training=training)

            for i in range(self.num_layers):
                x = self.enc_layers[i](x, training, mask)

            return x  # (batch_size, input_seq_len, d_model)


    # ### Decoder

    #  The `Decoder` consists of:
    # 1.   Output Embedding
    # 2.   Positional Encoding
    # 3.   N decoder layers
    #
    # The target is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the decoder layers. The output of the decoder is the input to the final linear layer.

    # In[ ]:


    class Decoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                     embedding, rate=0.1):
            super(Decoder, self).__init__()

            self.d_model = d_model
            self.num_layers = num_layers

            self.embedding = embedding
            self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)

            self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, enc_output, training,
                 look_ahead_mask, padding_mask):
            seq_len = tf.shape(x)[1]
            attention_weights = {}

            x = self.embedding(x)  # (batch_size, target_seq_len, d_model). The targets.
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]

            x = self.dropout(x, training=training)

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                       look_ahead_mask, padding_mask)

                attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
                attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

            # x.shape == (batch_size, target_seq_len, d_model)
            return x, attention_weights


    # ## Create the Transformer

    # Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the input to the linear layer and its output is returned.

    # In[ ]:

    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask


    class Transformer(tf.keras.Model):
        def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate=0.1):
            super(Transformer, self).__init__()

            self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

            self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                   vocab_size, self.embedding, rate)

            self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                                   vocab_size, self.embedding, rate)

            self.final_layer = tf.keras.layers.Dense(vocab_size)

        def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
            enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, dec_padding_mask)

            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

            return final_output, attention_weights

    def model(prompt, predicted, is_training):
        """Constructs the ResNet model given the inputs."""

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(prompt, predicted)

        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        prompt_length = 32

        # vocab_size = int(tokenizer.vocab_size + 2)
        dropout_rate = 0.1

        transformer = Transformer(num_layers, d_model, num_heads, dff, vocab_size, dropout_rate)
        predictions, _ = transformer(prompt, predicted[:, :-1],
                                     is_training,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        return predictions

    return model
