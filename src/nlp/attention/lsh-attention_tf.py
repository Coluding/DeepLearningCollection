import tensorflow as tf


class LSHAttention(tf.keras.Model):
    def __init__(self, bucket_size=5, n_hashes=1):
        super().__init__()
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

    def call(self, query, key, value, causal_masking=False):
        R = tf.random.normal((tf.shape(query)[0], tf.shape(query)[-1], self.bucket_size // 2))
        # xR is of shape batch x sequence_length x buckets // 2
        xR = tf.matmul(query, R)
        concat_xR = tf.concat([xR, -xR], axis=-1)
        # shape of batch x sequence_length x buckets

        buckets = tf.math.argmax(concat_xR, axis=-1)

        sticker = tf.argsort(buckets)
        undo_sort = tf.argsort(sticker)

        sorted_query = tf.gather(query, sticker, axis=1, batch_dims=1)
        sorted_value = tf.gather(value, sticker, axis=1, batch_dims=1)

        chunked_query = tf.stack(tf.split(sorted_query, self.bucket_size, 1), 1)
        chunked_value = tf.stack(tf.split(sorted_value, self.bucket_size, 1), 1)

        return sticker

