# Simple Character-Level RNN Example

This script trains a character-level GRU (Gated Recurrent Unit) model on a text file. After training, it can generate new text based on a user-provided prompt.

## Prerequisites

- Python 3.x
- PyTorch
- A text file named `story.txt` in the same directory as `main.py`. This file will be used as the training data.

## Parameters

The script accepts the following command-line arguments:

-   `-bs`, `--batch_size`: Batch size for training.
    -   *Default*: `32`
-   `-sl`, `--seq_len`: Sequence length for training (how many characters the model looks at in each step).
    -   *Default*: `10`
-   `-ml`, `--max_len`: Maximum length of the generated text.
    -   *Default*: `500`
-   `-e`, `--epochs`: Number of training epochs.
    -   *Default*: `30`

## Running the script

You can run the script using `uv` (or `python` if you manage your environment differently).

### Example Command:

```bash
uv run main.py -bs 32 -sl 100 -ml 1000 -e 10
```

This command will:
- Start training with a batch size of `32`.
- Use a sequence length of `100` characters.
- Set the maximum generation length to `1000` characters.
- Train the model for `10` epochs.

After training, the script will prompt you to enter text. The model will then attempt to continue your prompt. Enter a blank line to quit.