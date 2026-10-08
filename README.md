# DetAng

DetAng is an mBERT based model used to detect anglicisms in sentences.

Tokenization:
WordPiece tokenizer trained on a dataset created by myself. 
Pre-tokenizer splits text and WordPiece effectively splits anglicisms into sub-word token, using some special-case tokens.

Model architecture:
Transformer based encoder designed for speed and simplicity.
A linear classification layer is placed on top of the transformer encoder outputs to assign a class label to each individual token (e.g., Anglicism vs. Native word).
The model is randomly initialized and trained from scratch (Learning Rate: 5 * 10^(-4), 10 epochs) rather than relying on fine-tuning a pre-trained checkpoint.

Real use:
This project is still WIP. As of right now, the model is trained solely on Czech, making it the only language with sustainable results.
I plan on making it multilingual in the future, as well as experimenting with the model itself, making it more precise and faster.
