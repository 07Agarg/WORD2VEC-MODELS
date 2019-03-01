# WORD2VEC MODELS 

Word2Vec models, introduced by Mikolov Et.al are deep learning based models to compute and generate high quality distributed and continuous dense vector representations of words which capture a large number of precise synctactic and semantic similarity. 

C-BOW and Skip Gram Models are two different architectures of Word2Vec to create the word embedding representations. 

## CBOW Architecture:

Given the surrounding context words, the model tries to predict the current word. In the implementation in this repo, two context words on each side of current word are used. Input to the model will be [w(t-2), w(t-1), w(t+1), w(t+2)] and output will be w(t). These models are several times faster to train as compared to skip gram model. Gives better accuracy for frequent words. 

For Example: 
For a sentence, "The quick brown fox jumps over the lazy dog."
### Training Samples : 
- Input 						Output
- [quick, brown]				the
- [the, brown, fox]				quick
- [the, quick, fox, jumps] 		brown
- [quick, brown, jumps, over]  	fox
- [brown, fox, over, the]		jumps
- [fox, jumps, the, lazy]		over
- [over, the, dog] 				lazy


## Skip Gram Architecture:
Given a word, the model tries to predict the context surrounding words. In the implementation of this repo, model tries to figure out two surrounding words on either side of current word. Input to the model will be w(t) and output will be [w(t-2), w(t-1), w(t+1), w(t+2)]. These models work well with small amount of training data, represents well even rare words or phrases. 

For Example:
For a sentence, "The quick brown fox jumps over the lazy dog."
### Training Samples:
- Input 				Output
- the 					brown

- quick					the
- quick					brown
- quick 				fox

- brown 				the
- brown 				quick
- brown 				fox
- brown 				jumps

- fox					quick
- fox 					brown
- fox 					jumps
- fox 					over


It seems CBOW performs better than the skip-gram. This is probably because the inputs are richer than in the skip-gram model. In other words, assuming the sentence, The quick brown fox jumps over the lazy dog, though the input, output tuples for skip gram were single input, single output (e.g. input:'quick',output:'brown'), there are multiple inputs for a single output in CBOW (i.e. input:['the', 'quick', 'fox', 'jumps'], output:'brown'). As you can see from the example, in a given instance CBOW knows brown occurs when [the, quick, fox, jumps] words are “collectively” present where skip-gram only knows that brown occurs around quick.