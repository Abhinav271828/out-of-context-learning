# out-of-context-learning

# scripts

## `autoencoder.py`

The file has the class for the autoencoder and the lightning class to train the autoencoder. 

The autoencoder is structured as a down-projection followed by an up-projection to reconstruct the strings in the grammar using MSE loss

```
class AutoEncoder
    parameters:
        - str_len: length of the strings
        - hiddens: a list with all hidden dimensions
    
    this will create an auto encoder with the following config

    str_len -> hidden_dims -> reversed hidden_dims -> str_len
```

And a training class for the AutoEncoder





## `data_gen.py`





## `model_def.py`

Definitions for our custom attention layer (modified code from huggingface)

We had to redefine attention because we need to get rid of softmax in the attention



## `transformer_with_auto_features.py`

This has the model and training classes that puts our custom self attention on top of the fully self supervised trained auto-encoder. 

The autoencoder is kept frozen 

The training happens to do the actual task. 

## dataset_scripts

### `auto_encoder_data.py`

dataset class for the autoencoding task. 

# `find_lr.py`

A function to find the learning rate to be used in the construction of the  MHA object the authors present to do linear regression as a setup.


# `construction.py`

Constructs the MHA using the paper's closed form formula.

# `dataset.py`




# `test_encodings.py`

Tests a trained autoencoder to check if a logistic regression can be learnt on top of those embeddings to do the task of string membership

# `visualize.py`

Plotting code for the weight matrices and their comparisons 
