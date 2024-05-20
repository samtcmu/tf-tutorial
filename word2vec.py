import tensorflow as tf

def CreateVocabulary(tokens):
    vocabulary = {'<pad>': 0}  # Assign ID 0 to a padding token.
    next_id = 1
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = next_id
            next_id += 1
    return vocabulary


def VectorizeSentence(sentence):
    tokens = sentence.lower().split()
    vocabulary = CreateVocabulary(tokens)
    inverse_vocabulary = {token_id: token for token, token_id in vocabulary.items()}
    return [vocabulary[token] for token in tokens], vocabulary, inverse_vocabulary


def GenerateSkipGrams(sentence_vector, vocabulary, window_size=2):
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sentence_vector,
          vocabulary_size=len(vocabulary),
          window_size=window_size,
          negative_samples=0)
    return positive_skip_grams


def GenerateNegativeSamples(skip_grams, vocabulary, inverse_vocabulary, num_negative_samples=5, seed=42):
    for target_token, context_token in skip_grams:
        context_class = tf.reshape(tf.constant(context_token, dtype="int64"), (1, 1))
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=num_negative_samples,
            unique=True,
            range_max=len(vocabulary),
            seed=seed,
            name="negative_sampling"
        )
        squeezed_context_class = tf.squeeze(context_class, 1)
        context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*num_negative_samples, dtype="int64")
        print(f"target_index    : {target_token}")
        print(f"target_word     : {inverse_vocabulary[target_token]}")
        print(f"context_indices : {context}")
        print(f"context_words   : {[inverse_vocabulary[c.numpy()] for c in context]}")
        print(f"label           : {label}")
        break


def main():
    sentence = "The wide road shimmered in the hot sun"
    sentence_vector, vocabulary, inverse_vocabulary = VectorizeSentence(sentence)
    sentence_skip_grams = GenerateSkipGrams(sentence_vector, vocabulary, window_size=2)
    for target_token, context_token in sentence_skip_grams:
        t = inverse_vocabulary[target_token]
        c = inverse_vocabulary[context_token]
        print(f"({target_token:d}, {context_token:d}): ({t:s}, {c:s})")
    GenerateNegativeSamples(sentence_skip_grams, vocabulary, inverse_vocabulary, num_negative_samples=4)


if __name__ == "__main__":
    main()

