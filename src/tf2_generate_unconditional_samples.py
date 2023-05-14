#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
from model import HParams

import model, sample, encoder

def sample_model(
    model_name='124M',
    seed=None,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Run the sample_model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
def sample_model(
    model_name='124M',
    seed=None,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    # Create model
    X = tf.zeros([batch_size, 1], dtype=tf.int32)
    past = None
    transformer = model.Transformer(hparams)
    result = transformer(X, past=past)
    output = result["logits"][:, -1, :] / temperature

    # Sample from logits
    output = tf.random.categorical(output, num_samples=1)
    output = tf.squeeze(output, 1)
    out = tf.Variable(np.zeros((batch_size, 0), dtype=np.int32), trainable=False)

    # Create a checkpoint and restore from the checkpoint
    transformer_model = model.Transformer(hparams)
    ckpt = tf.train.Checkpoint(model=transformer_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(models_dir, model_name), max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    # Run sampling
    generated = 0
    while nsamples == 0 or generated < nsamples:
        samples = []
        for _ in range(length):
            sample_out = out[:, -hparams.n_ctx:]
            o = tf.concat([sample_out, output], axis=-1)
            out = tf.concat([out, o], axis=1).numpy()

            if generated % batch_size == 0:
                samples.append(out)
                out = tf.Variable(np.zeros((batch_size, 0), dtype=np.int32), trainable=False)

            generated += len(samples)

        for out in samples:
            text = enc.decode(out)
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

if __name__ == '__main__':
    fire.Fire(sample_model)