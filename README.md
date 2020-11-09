# Stochastic POS Tagger: HMM-based
PoS tagging consists in assigning a tag to each word of a corpus. The choice of which tagset to use depends on the 
language/application. The input is a string of words and a tagset from use, the output instead is the 
association of the most appropriate tag to each word.
To better understand what a pos tagger of this type is, I recommend this reading: 
https://www.mygreatlearning.com/blog/pos-tagging/.

The datasets were taken from the web site https://universaldependencies.org/

## Implementation
In this project I have tried different types of training set processing by changing the way of treating words and 
saving the odds.

#### Storing words methods:
- **1:** all words are stored in lowercase;
- **2:** all words are stored as they are read in the training set without doing any operation.

#### Storing probabilities methods:
- **1:** P (w<sub>i</sub> | t<sub>i</sub>) for a word w_i never encountered with the tag t<sub>i</sub>
 in the training set is set to 0. The same is true for the probabilities P(t<sub>i</sub> |  t<sub>i</sub> - 1) of the 
 tag sequences never encountered; 
 - **2:** a word encountered only with a tag t<sub>i</sub> ∈ I = {NOUN, ADJ, VERB, ADV, PROPN}, will take 
t<sub>i</sub> with a probability of 99% and t<sub>j</sub> ∈ I \ t<sub>i</sub> with a probability of 0.25%. A similar 
argument holds for the probabilities of tag sequences P (t<sub>i</sub> | t<sub>i</sub> - 1) as explained above.

#### Smoothing methods:
 - **1:** P(word|PROPN) = 1 and P(word|tag != PROPN) = 0
 - **2:** P(word|t<sub>i</sub>) = P(t<sub>i</sub>) where
<img src="http://www.sciweavers.org/tex2img.php?eq=P%28t_i%29%3D%5Cfrac%7Bf%28t_i%29%7D%7B%5Csum_%7Bj%3D1%7D%5En%20f%28t_j%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit="/>     
f(t<sub>k</sub>) = frequency of the tag t<sub>k</sub>
- **3:** P (word|t<sub>i</sub>) = P (W<sup>1</sup> |t<sub>i</sub>) where W<sup>1</sup> is the set made up of all
those words that appear only once in the training set. The probability distribution is calculated by taking into account
only words that appear only once in the training set and are assigned to new word word.

## Run
To run this project you have to set `--training-set` and `--test-set` parameters that indicate the path of the two 
dataset. You can set also the parameter `--validation-set` that will be added to the training set.
Other parameters that you can set are:
- `--storing-data`: words storing method;
- `--storing-prob`: probability storing method;
- `--smoothing`: smoothing method for the unknown words.

Example of command:

```python3 app.py --training-set=ud-treebanks-v2.3/UD_English-LinES/en_lines-ud-train.conllu --validation-set=ud-treebanks-v2.3/UD_English-LinES/en_lines-ud-dev.conllu --test-set=ud-treebanks-v2.3/UD_English-LinES/en_lines-ud-test.conllu```
