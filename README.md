# jaxpt2

# todo

-   eval
    -   proper sampling
-   logging
-   saving

# ideas

-   "standard" model sizes
-   robust tokenization/dataloading
    -   huggingface datasets?
-   attention
    -   efficient attention
    -   recurrence
-   training
    -   AMP
    -   checkpointing
    -   accumulation
    -   revnets
    -   memory swapping
-   post-training
    -   distillation
    -   pruning/lottery ticket
    -   quantization
-   parallelism
    -   model
    -   pipeline
    -   deepspeed/zero/offload/1bitadam
    -   fusedadam
-   encoder/decoder/encoder-decoder
-   weight sharing
-   nfnet
-   alternatives to autoregressive LM
-   multimodal
-   vqvae/gans
-   optimizer

# done

-   tie input/output embedding weights
    -   make a custom embedding/dense layer
    -   turns out flax embeddings have a `.attend()` method just for this

# tokenizer

```python
from tokenizers import ByteLevelBPETokenizer

files = [f"./wikitext-2-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files, vocab_size=32768)
tokenizer.save('./tokenizer.json')
tokenizer = Tokenizer.from_file('./tokenizer.json')

tokenizer.encode('Hello, my name is bilal').tokens
tokenizer.decode(tokenizer.encode('Hello, my name is bilal').ids)
```
