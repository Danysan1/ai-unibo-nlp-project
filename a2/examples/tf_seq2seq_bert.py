"""
This was tested with:
tensorflow==2.6
tensorflow-gpu==2.6
tensorflow-addons==0.16.1
transformers==4.18.0
Keras==2.6.0

Note 1: Simple adaptation of tf_seq2seq_lstm.py script
Note 2: make sure Keras and Tensorflow versions match!

"""

import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from transformers import TFAutoModel, AutoTokenizer

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class MyTrainer(object):
    """
    Simple wrapper class

    train_op -> uses tf.GradientTape to compute the loss
    batch_fit -> receives a batch and performs forward-backward passes (gradient included)
    """

    def __init__(self, encoder, decoder, max_length):
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)

    @tf.function
    def compute_loss(self, logits, target):
        loss = self.ce(y_true=target, y_pred=logits)
        mask = tf.logical_not(tf.math.equal(target, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_mean(loss)

    @tf.function
    def train_op(self, inputs):
        with tf.GradientTape() as tape:
            encoder_output, encoder_h, encoder_s = self.encoder({'input_ids': inputs['encoder_input_ids'],
                                                                 'attention_mask': inputs['encoder_attention_mask']})

            decoder_input = inputs['decoder_target'][:, :-1]  # ignore <end>
            real_target = inputs['decoder_target'][:, 1:]  # ignore <start>

            decoder.attention.setup_memory(encoder_output)

            decoder_initial_state = self.decoder.build_initial_state(decoder.batch_size, [encoder_h, encoder_s])
            predicted = self.decoder({'input_ids': decoder_input,
                                      'initial_state': decoder_initial_state}).rnn_output

            loss = self.compute_loss(logits=predicted, target=real_target)

        grads = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        return loss, grads

    @tf.function
    def batch_fit(self, inputs):
        loss, grads = self.train_op(inputs=inputs)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))
        return loss

    # @tf.function
    def generate(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        encoder_output, encoder_h, encoder_s = self.encoder({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

        start_tokens = tf.fill([batch_size], output_tokenizer.word_index['<start>'])
        end_token = output_tokenizer.word_index['<end>']

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.wrapped_decoder_cell,
                                                    sampler=greedy_sampler,
                                                    output_layer=self.decoder.generation_dense,
                                                    maximum_iterations=self.max_length)
        self.decoder.attention.setup_memory(encoder_output)

        decoder_initial_state = self.decoder.build_initial_state(batch_size, [encoder_h, encoder_s])
        decoder_embedding_matrix = self.decoder.embedding.variables[0]
        outputs, _, _ = decoder_instance(decoder_embedding_matrix,
                                         start_tokens=start_tokens,
                                         end_token=end_token,
                                         initial_state=decoder_initial_state)
        return outputs

    def translate(self, generated):
        return output_tokenizer.sequences_to_texts(generated.sample_id.numpy())


class Encoder(tf.keras.Model):

    def __init__(self, model_name, decoder_units):
        super(Encoder, self).__init__()
        self.model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        self.reducer = tf.keras.layers.Dense(decoder_units)

    def call(self, inputs, training=False, **kwargs):
        model_output = self.model(inputs)
        all_outputs = model_output[0]
        pooled_output = model_output[1]
        pooled_output = self.reducer(pooled_output)
        return all_outputs, pooled_output, pooled_output


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, max_sequence_length, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()

        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size

        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim)
        self.decoder_lstm_cell = tf.keras.layers.LSTMCell(self.decoder_units)

        self.attention = tfa.seq2seq.BahdanauAttention(units=self.decoder_units,
                                                       memory=None,
                                                       memory_sequence_length=self.batch_size * [max_sequence_length])

        self.wrapped_decoder_cell = tfa.seq2seq.AttentionWrapper(self.decoder_lstm_cell,
                                                                 self.attention,
                                                                 attention_layer_size=self.decoder_units)

        self.generation_dense = tf.keras.layers.Dense(vocab_size)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.decoder = tfa.seq2seq.BasicDecoder(self.wrapped_decoder_cell,
                                                sampler=self.sampler,
                                                output_layer=self.generation_dense)

    def build_initial_state(self, batch_size, encoder_state):
        initial_state = self.wrapped_decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        initial_state = initial_state.clone(cell_state=encoder_state)
        return initial_state

    def call(self, inputs, training=False, **kwargs):
        input_ids = inputs['input_ids']
        input_emb = self.embedding(input_ids)
        decoder_output, _, _ = self.decoder(input_emb,
                                            initial_state=inputs['initial_state'],
                                            sequence_length=self.batch_size * [self.max_sequence_length - 1])
        return decoder_output


if __name__ == '__main__':
    model_name = 'prajjwal1/bert-tiny'

    # tf.config.run_functions_eagerly(True)

    # Sample
    input_sample = [
        "hello there how is it going",
        "this assignment is hellish"
    ]
    output_sample = [
        "<start> it is going well <end>",
        "<start> I agree <end>"
    ]

    batch_size = len(input_sample)

    # Input
    input_tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_inputs = input_tokenizer(input_sample, return_tensors='tf', padding=True)
    input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
    max_input_length = input_ids.shape[-1]

    # Output
    output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<UNK>')
    output_tokenizer.fit_on_texts(output_sample)

    output_vocab_size = len(output_tokenizer.word_index) + 1

    encoded_output_sample = output_tokenizer.texts_to_sequences(output_sample)
    max_output_length = max([len(item) for item in encoded_output_sample])

    max_sequence_length = max(max_input_length, max_output_length)

    encoded_output_sample = tf.keras.preprocessing.sequence.pad_sequences(encoded_output_sample,
                                                                          padding='post',
                                                                          maxlen=max_sequence_length)

    # Test encoder
    encoder = Encoder(model_name=model_name,
                      decoder_units=16)
    encoder_output, encoder_h, encoder_s = encoder({'input_ids': input_ids,
                                                    'attention_mask': attention_mask})
    print(f'{encoder_output.shape} - {encoder_h.shape} - {encoder_s.shape}')

    # Test decoder
    decoder = Decoder(vocab_size=output_vocab_size,
                      embedding_dim=50,
                      decoder_units=16,
                      batch_size=batch_size,
                      max_sequence_length=max_sequence_length)
    decoder.attention.setup_memory(encoder_output)
    initial_state = decoder.build_initial_state(batch_size, [encoder_h, encoder_s])

    decoder_sample_batch = {
        'input_ids': tf.convert_to_tensor(encoded_output_sample, tf.int32),
        'initial_state': initial_state
    }
    sample_decoder_outputs = decoder(decoder_sample_batch).rnn_output
    print(f'{sample_decoder_outputs.shape}')

    # Training
    trainer = MyTrainer(encoder=encoder,
                        decoder=decoder,
                        max_length=max_sequence_length)
    epochs = 100
    for epoch in tqdm(range(epochs)):
        batch = {
            'encoder_input_ids': input_ids,
            'encoder_attention_mask': attention_mask,
            'decoder_target': encoded_output_sample
        }
        loss = trainer.batch_fit(batch)
        print(f'Loss - {loss}')

        generated = trainer.generate(input_ids=input_ids,
                                     attention_mask=attention_mask)
        translated = trainer.translate(generated)
        print(f'Translated - {translated}')
