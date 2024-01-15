LLAMA_FEEDBACK = """[INST]
<<SYS>>
Suppose you are a professional translator and good at providing concise assessments for machine translation results.
<</SYS>>
I will provide you a Chinese sentence X and its English translation Y, please assess it in a concise way.
Requirements:
1. You should first provide an overall evaluation.
2. Following that, if there are any errors, please point them out and explain. If there are no errors, you do not need to provide an explanation.
3. When explaining errors, please specify the error type, the corresponding segment in the Chinese sentence X, the corresponding segment in the translation Y, the reason for the error, and the corrected translation.
4. For multiple errors, you should address them separately.
5. [Important!] Try to pinpoint the smallest segments containing errors and explain them, avoiding cases where the error encompasses the entire sentence.
6. Carefully read the original text and the translation to identify all translation errors.
7. For errors related to articles, singular/plural, and similar issues that do not significantly affect semantic comprehension, there is no need to point them out.
8. Your response should be in English.

Now, please assess the following translation:

Chinese: <srctext>
English: <tgttext>
[/INST]
Assessment:
"""

LLAMA_SENTENCE_ANALYSIS = """[INST]
<<SYS>>
Suppose you are a language expert of Chinese and English.
<</SYS>>
Given a sentence X, please point out its topic, domain and style.
Input:
X: <src_text>
[/INST]
Output:
"""

LLAMA_CASE_GENERATION = """[INST]
<<SYS>>
Support you are a language expert of Chinese and English.
<<SYS>>
Given a topic, a domain and a style, as well as a bilingual word pair, please generate a pair of parallel sentence that adhere to the given topic, domain and style. They should also contain the given word pair.
Input:
<domain_topic_style>
Word Pair: <word_pair>
[/INST]
Output:
"""

LLAMA_WORD_ANALOGY = """"""
