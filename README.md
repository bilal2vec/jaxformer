# jaxformer

# todo

-   model parameters
-   logging
-   saving
-   https://github.com/google-research/google-research/tree/master/flax_models/t5x

# done

-   tie input/output embedding weights
    -   make a custom embedding/dense layer
    -   turns out flax embeddings have a `.attend()` method just for this
-   eval
    -   proper sampling

# tokenizer

```python
from tokenizers import Tokenizer, ByteLevelBPETokenizer

files = [f"./wikitext-2-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files, vocab_size=32768)
tokenizer.save('./tokenizer.json')
tokenizer = Tokenizer.from_file('./tokenizer.json')

tokenizer.encode('Hello, my name is bilal').tokens
tokenizer.decode(tokenizer.encode('Hello, my name is bilal').ids)
```
