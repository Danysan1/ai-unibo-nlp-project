"""
This was tested with:
tensorflow==2.6
tensorflow-gpu==2.6
transformers==4.18.0
Keras==2.6.1

Note 1: make sure Keras and Tensorflow versions match!
Note 2: Not sure if TFEncoderDecoderModel is available in previous transformers versions

"""

from transformers import TFEncoderDecoderModel, AutoTokenizer
import tensorflow as tf
from tqdm import tqdm
from copy import deepcopy

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

    def __init__(self, keras_model):
        self.keras_model = keras_model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-05)

    @tf.function
    def compute_loss(self, inputs):
        loss = self.keras_model(inputs=inputs)
        return tf.reduce_mean(loss)

    @tf.function
    def train_op(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs=inputs)

        grads = tape.gradient(loss, self.keras_model.trainable_variables)
        return loss, grads

    @tf.function
    def batch_fit(self, inputs):
        loss, grads = self.train_op(inputs=inputs)
        self.optimizer.apply_gradients(zip(grads, self.keras_model.trainable_variables))
        return loss


class MyModel(tf.keras.Model):
    """
    Custom keras model that wraps the TFEncoderDecoderModel
    """

    def __init__(self, model_name, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.model_name = model_name

        # tie_encoder_decoder to share weights and half the number of parameters
        self.model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name,
                                                                           encoder_from_pt=True,
                                                                           decoder_from_pt=True,
                                                                           tie_encoder_decoder=True)

    def call(self, inputs, **kwargs):
        loss = self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['input_attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['labels_mask'],
                          labels=inputs['labels']).loss
        return loss

    def generate(self, **kwargs):
        return self.model.generate(decoder_start_token_id=self.model.config.decoder.pad_token_id,
                                   **kwargs)


if __name__ == '__main__':
    """
    Example main
    """

    model_name = 'distilroberta-base'
    input_sample = [
        "hello there how is it going",
    ]
    output_sample = [
        "it is going well",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MyModel(model_name=model_name)

    model.model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.model.config_eos_token_id = tokenizer.sep_token_id
    model.model.config.pad_token_id = tokenizer.pad_token_id
    model.model.config.vocab_size = model.model.config.encoder.vocab_size

    trainer = MyTrainer(keras_model=model)

    input_values = tokenizer(input_sample, add_special_tokens=False, padding=True)
    input_ids, input_attention_mask = input_values['input_ids'], input_values['attention_mask']
    label_values = tokenizer(output_sample, padding=True)
    labels, labels_mask = label_values['input_ids'], label_values['attention_mask']

    max_length = len(labels[0])

    masked_labels = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in labels]

    epochs = 20
    for epoch in tqdm(range(epochs)):
        batch = {'input_ids': tf.convert_to_tensor(input_ids, dtype=tf.int32),
                 'input_attention_mask': tf.convert_to_tensor(input_attention_mask, dtype=tf.int32),
                 'labels': tf.convert_to_tensor(masked_labels, dtype=tf.int32),
                 'decoder_input_ids': tf.convert_to_tensor(deepcopy(labels), dtype=tf.int32),
                 'labels_mask': tf.convert_to_tensor(labels_mask, dtype=tf.int32)
                 }
        loss = trainer.batch_fit(inputs=batch)
        print(f'Epoch {epoch} -- Loss {loss}')

        # You can play with generation arguments to enforce
        #  beam search
        #  repetition penalty
        #  other sampling approaches
        generated = trainer.keras_model.generate(input_ids=tf.convert_to_tensor(input_ids, dtype=tf.int32),
                                                 max_length=max_length,
                                                 repetition_penalty=3.,
                                                 min_length=5,
                                                 no_repeat_ngram_size=3,
                                                 early_stopping=True,
                                                 num_beams=4
                                                 )
        generated = tokenizer.batch_decode(generated, skip_special_tokens=True)
        print(f'Generated: {generated}')