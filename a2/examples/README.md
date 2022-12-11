## Question Answering with CoQA

Copied from https://gitlab.com/nlp-unibo/nlp-unibo-material/-/blob/master/2022-2023/Assignment%202/README.md

## General

You have freedom of choice concerning model definition and deep learning libraries.
Here's a brief list of allowed actions:

* Bert2Bert
* Seq2Seq w/ Bert
* Bert span selection and LSTM generator (a bit more complex)

## Bert2Bert

The ```tf_bert_generation.py``` script shows how to load and use BertGeneration models from <a href="https://huggingface.co/docs/transformers/model_doc/bert-generation">Huggingface</a>.

## Seq2Seq w/ Bert

The ```tf_seq2seq_bert.py``` script is a simple adaptation of ```tf_seq2seq_lstm.py```.
The latter script shows how to define a simple LSTM-based seq2seq architecture (code is largely adapted from <a href="https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt">Tensorflow's documentation</a>).

## Bert span selection and LSTM generator

It is a variant of Seq2Seq w/ BERT where the encoder input is the span selected by a Bert-based model.
Here' s a quick schema:
```
    span = span_model(inputs)                               # 1st model
    encoder_outputs, encoder_h, encoder_s = encoder(span)   # 2nd model
    generated = decoder(encoder_outputs)                    # 3rd model
```

## FAQ (cont'd)

### [Bert2Bert] The model is quite heavy, what can I do?

* Tie encoder-decoder weights -> ```Model.from_pretrained(..., tie_encoder_decoder=True)```
* Reduce input length -> maximum length might be too much (try quantiles!)
* Reduce batch size

### [Bert2Bert] The model is quite slow...

* Run a notebook instance for each model -> save & load stuff to speed up the whole process!
* These models are a bit heavy, ~2/3 hours of training are normal

### [Bert2Bert] My model is not generating good stuff...

There are a lot of options for inference to account for this issue.
Check the ```model.generate()``` signature for more details!

* Beam search
* Top-k sampling
* Repetition penalty
* Nucleus sampling

### [Seq2Seq w/ Bert] Can I freeze the encoder model?

Sure you can! This is a legit solution.
You can also add additional layers to your encoder (e.g., MLP) to increase its complexity.

There are a lot of options for inference to account for this issue.
Check the ```model.generate()``` signature for more details!

* Beam search
* Top-k sampling
* Repetition penalty
* Nucleus sampling

### [Seq2Seq w/ Bert] Can I use BeamSearch?

Sure you can! Please check the last section of this tensorflow <a href="https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt#use_tf-addons_beamsearchdecoder">tutorial</a>.

### [General] Should I apply a sliding window to avoid truncation?

Applying a sliding window to build multiple samples from a given one is a widely adopted technique.
However, the main problem relies in building negative samples (i.e., the ones with no ground-truth answer).

For the sake of the assignment, we just request you to truncate long input samples according to your model requirements.

In any case, we do not forbid using sliding windows. Here are some tips:

- Negative samples: those that do not have the ground-truth rationale (R) in the extracted view of P.
- Negative samples label: these samples are unaswerable samples (in this case, you can avoid filtering out original unaswerable QA pairs)
- Undersample negative samples to get a balanced dataset: you will get a lot of negative samples with this technique. Avoid having a strongly unbalanced (and large) dataset

--

<u><b>We don't expect to obtain high quality results! This is not the aim of the assignment</b></u>