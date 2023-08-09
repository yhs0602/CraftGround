import sentencepiece as spm


if __name__ == "__main__":
    spm.SentencePieceTrainer.train(
        input="fishing.txt",
        model_prefix="m",
        vocab_size=800,
        user_defined_symbols=["foo", "bar"],
    )
