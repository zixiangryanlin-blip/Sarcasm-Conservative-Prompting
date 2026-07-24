# Rethinking Sarcasm Detection: System 1 Prompting with Conservative Enhancement

This repository serves as a complete archive of the code and data used to investigate whether LLMs perform sarcasm detection better with System 1 (intuitive) or System 2 (deliberative) prompts, and to explore the improvement achieved by Conservative Prompting.

What's Included in This Repository:

1. **Code folder:** Python scripts used to interface with the LLMs;
2. **Raw Data Counts:** The raw classification counts (True Positive, False Positive, True Negative, False Negative) for six LLMs across four prompts;
3. **Data Analysis:** Statistical evaluation and matrix.

**Abstract:** Sarcasm detection is a core problem in natural language processing (NLP), requiring models to infer the intended meaning that cannot be captured by lexical mapping. This short paper investigates whether large language models (LLMs) perform sarcasm detection better in a fast, intuitive System 1 prompt or a slower, deliberative System 2 prompt. Using the iSarcasmEval dataset, we compare Zero-Shot (System 1) with Zero-Shot-Chain-of-Thought (Zero-Shot-CoT) (System 2) promptings across six LLMs. Additionally, we introduce Conservative Prompting, a threshold-based strategy that biases models toward non-sarcastic predictions unless strong evidence supports sarcasm. Experimental results show that Zero-Shot prompting outperforms Zero-Shot-CoT in accuracy and F1-score, supporting the claim that System 1 surpasses System 2 prompts in sarcasm detection. Furthermore, Conservative Prompting significantly improves performance by increasing precision and reducing false positives. Together, these findings contribute a novel perspective to the sarcasm detection task and promote safer human-computer interaction.
