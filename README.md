## A barebones transformer implementation

Many transformer implementations feature extreme amounts of optimization and abstraction at the cost of modularity and readability. This repo tries to fill the lack of usable and easy to read transformer code.

A simple example is provided in the form of a Jupyter Notebook; all information regarding the transformer itself is present within the files.

MIT license, modify as you desire. Use it for something cool? Send me a message.

```
* EXAMPLES/
    IWSLT-De-En.ipynb - Basic De->En translation 
    IWSLT-En-MLM.ipynb - Basic MLM on the En portion of IWSLT2016
    
* model/
  - EncoderDecoder.py - contains TransformerEncoder and TransformerDecoder code
  - Layers.py - Contains code for Linear and Embedding layers with proper initialization
  - LearnedPositionalEmbedding - Contains code for the learned positional encoding
  - TransformerLayer - Contains TransformerEncoderLayer and TransformerDecoderLayer code
  - opt.py - Contains optimizer wrapper
  - transformers.py - Contains the fully Enc-Dec transformer
  - utils.py - Contains objects useful for handling data
```