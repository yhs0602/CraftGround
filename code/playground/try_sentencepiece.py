import sentencepiece as spm

if __name__ == "__main__":
    s = spm.SentencePieceProcessor(model_file="m.model")
    print(s.encode("This is a test"))
    print(
        s.encode(
            "The fishing rod can occasionally catch treasure or junk instead of fish."
        )
    )
    print(
        s.encode(
            "The bobber floats up to the water's surface even if the fishing rod is used from underwater"
            " unless it hooks onto a mob such as squid or is obstructed by a block."
        )
    )
    print(
        s.encode(
            "The player can catch fish even if occupying the same block space as the bobber."
        )
    )
    print(s.encode("This does not vary depending on enchantments."))
    print(s.encode("Enchant a fishing rod."))
