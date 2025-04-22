# Seminar 13: LLMs with Transformers & LangChain

This folder contains materials for the thirteenth seminar in our **Deep Generative Models** course. In this session we shift to **large‑language‑model operations**—covering inference with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), custom model/tokenizer workflows, finetuning via the Trainer API, and stateful agent construction with [LangChain](https://python.langchain.com/docs/introduction/) & [LangGraph](https://langchain-ai.github.io/langgraph/).

## Seminar Overview  

- **Topic**: Practical LLM Deployment with Hugging Face Transformers and LangChain  
- **Objectives**  
  - **Instant Inference with Pipelines**: Use the one‑line `pipeline()` abstraction for sentiment analysis, automatic speech recognition (ASR), and vision tasks.  
  - **Custom Models & Tokenizers**: Swap in domain‑specific checkpoints (e.g., `nlptown/bert-base-multilingual-uncased-sentiment`) and multilingual tokenizers; inspect token IDs and attention masks.  
  - **Finetuning with the Trainer API**: Configure `TrainingArguments`, build dynamic padding with `DataCollatorWithPadding`, and launch a reproducible experiment on the Rotten Tomatoes dataset.  
  - **LLM Agents with LangChain + LangGraph**: Wrap a `Qwen2‑0.5B-Instruct` text‑generation pipeline, add memory, build prompt templates, checkpoint conversation graphs, and trim context to stay within token budgets.  
  - **Hands‑on Implementation**: Every demo is executable in the companion notebook—no external credentials required.

## Key Concepts  

1. **Transformers Pipelines**  
2. **Custom Tokenization & Model Loading**  
3. **AutoClasses & Low‑Level Forward Pass**  
4. **Trainer API for Finetuning**  
5. **LangChain Integration**  
6. **LangGraph for Persistent Sessions**  
7. **Prompt Engineering & Trimming**  

## Useful Links  

### Code & Docs
- **Transformers API**: <https://huggingface.co/docs/transformers/index>  
- **Trainer API**: <https://huggingface.co/docs/transformers/en/trainer>  
- **LangChain Docs**: <https://python.langchain.com/>  
- **LangGraph Guide**: <https://python.langchain.com/docs/langgraph/>  