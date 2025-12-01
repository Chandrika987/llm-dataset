from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

CORPUS = r"C:\Users\chand\OneDrive\Desktop\llm\challenge2\corpus.txt"

tokenizer = Tokenizer(models.BPE())

# FIX: disable automatic leading space
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train(files=[CORPUS], trainer=trainer)

tokenizer.save("custom_tokenizer.json")
print("Tokenizer retrained WITHOUT prefix space!")
