from tensorflow.keras.preprocessing.text import Tokenizer
import inspect

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words= 100)
tokenizer.fit_on_texts(sentences)

# print(tokenizer.__dict__)
print(inspect.getmembers(tokenizer, predicate=inspect.ismethod))