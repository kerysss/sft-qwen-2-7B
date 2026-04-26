# sft-qwen-2-7B
微调小说续写模型（2026.02 - 2026.03）

- **目标**：基于《白鹿原》小说数据，微调Qwen-2-7B实现风格化续写。

- **方法**：使用hugging face上的白鹿原数据集，自定义对话模板，做tokenizer；使用LoRA + DeepSpeed ZeRO-2进行高效微调；使用vllm部署，体验对比商用大模型发现差很多，意识到这样搞就是个小玩具。

- **技术栈**：Python、Transformers、DeepSpeed、LoRA/peft、Vllm、tokenizer、datasets

- **成果**：模型在连贯性和风格一致性上优于baseline，展示了从数据选用到模型微调到部署评估的完整pipeline能力。
