# sft-qwen-2-7B
微调小说续写模型（2026.02 - 2026.03）

- **目标**：基于《白鹿原》小说数据，微调Qwen-2-7B实现风格化续写。

- **方法**：使用hugging face上的白鹿原数据集，自定义对话模板，做tokenizer；使用LoRA + DeepSpeed ZeRO-2进行高效微调；使用vllm部署，体验对比商用大模型发现差很多，意识到这样搞就是个小玩具。

- **技术栈**：Python、Transformers、DeepSpeed、LoRA/peft、Vllm、tokenizer、datasets

- **成果**：模型在连贯性和风格一致性上优于baseline，展示了从数据选用到模型微调到部署评估的完整pipeline能力。

训练环境

[慧星云](https://portal.huixingyun.com/console/gpu-generic)

GPU：NVIDIA 4090 * 1

显存：24 G

CPU：12 核

内存：60 G

区：华东六区

镜像：image-gpu-pytorch_20250820

环境命令

conda install cuda -c nvidia # 安装cuda toolkit， 为deepspeed安装做准备
nvcc --version # 安装成功查看版本
export CUDA_HOME=$CONDA_PREFIX # 配置环境变量
pip install deepspeed,bitsandbytes,transformers==5.5.4,accelerate==0.28.0 peft==0.9.0,datasets,vllm # 安装训练用的库

vi ds_config_zero2.json # deepspeed的执行的配置zero2
vi train_lora_deepspeed.py # 训练文件

vi deploy_lora_model.py # 部署使用文件
deepspeed --num_gpus=1 train_lora_deepspeed.py # 执行训练

ds_config_zero2.json

因为zero3和梯度检查点一起会有冲突，报错CheckpointError， DeepSpeed 与梯度检查点（Gradient Checkpointing）的经典冲突，尤其是在 ZeRO Stage 3 下与 LoRA 联合使用时非常常见。问题根源是 **重计算机制试图恢复一个已不存在的张量（形状从 [3584] 变成了 [0]）**，导致训练失败。

详细错误解释：

- **ZeRO Stage 3 的参数分区特性**：ZeRO-3 会将模型参数切分到不同 GPU 上，不归当前卡管的参数，其形状在本地会被处理成 `0`。当梯度检查点进行重计算时，就可能错误地读取了已经被 ZeRO-3 “清零” 的张量[](https://blog.gitcode.com/2ef400850ca524e1304e6f22e95e0839.html)[](https://blog.gitcode.com/2ef400850ca524e1304e6f22e95e0839.html)。
  
- **梯度检查点与 LoRA 的机制冲突**：梯度检查点在重计算时会跳过某些 hooks，可能导致 LoRA 更新失效，并最终引发元数据不匹配的错误
  

社区推荐的非常稳定的解决方案是将 DeepSpeed 配置文件中的 ZeRO Stage 从 3 改为 2。

{
 "bf16": {
 "enabled": true
 },
 "fp16": {
 "enabled": false
 },
 "zero_optimization": {
 "stage": 2,
 "offload_optimizer": {
 "device": "cpu",
 "pin_memory": true
 },
 "overlap_comm": true,
 "contiguous_gradients": true,
 "reduce_bucket_size": "auto"
 },
 "gradient_accumulation_steps": "auto",
 "gradient_clipping": "auto",
 "train_batch_size": "auto",
 "train_micro_batch_size_per_gpu": "auto",
 "wall_clock_breakdown": false
}

train_lora_deepspeed.py

数据集选择了hugging face上的silk-road/ChatHaruhi_NovelWriting，是一个对话式白鹿原小说补全的数据，把它用在Qwen/Qwen2-7B-Instruct模型需要修改格式为对话模板，然后再tokenizer，训练参数这里注意不同版本的transformer有变动，trainer训练器的参数也有这个问题，比如老版trainer不需要tokenizer参数。

```
#!/usr/bin/env python

# train_lora_deepspeed.py

import torch
from transformers import (
 AutoModelForCausalLM,
 AutoTokenizer,
 TrainingArguments,
 Trainer,
 DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import deepspeed

def main():
 # 配置参数
 model_name = "Qwen/Qwen2-7B-Instruct"
 output_dir = "/context/sample"
 dataset_name = "silk-road/ChatHaruhi_NovelWriting" # 替换为实际数据集
# 1. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False  # 训练时禁用KV cache
)

# 2. 启用梯度检查点（以时间换显存）
model.gradient_checkpointing_enable()

# 3. 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 数据准备
dataset = load_dataset(dataset_name, split="train")

def format_instruction(example):
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['context']}<|im_end|>\n<|im_start|>assistant\n{example['target']}<|im_end|>"
    return {"text": prompt}

dataset = dataset.map(format_instruction)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,    # 7B模型，24G显存建议为1，可尝试2
    gradient_accumulation_steps=8,    # 模拟更大的batch size: 1*8=8
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,                       # 若GPU支持bf16，设为False，并在model加载时指定
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    deepspeed="ds_config_zero3.json",  # DeepSpeed配置
    optim="paged_adamw_8bit",         # 使用8bit优化器节省显存
    report_to="none",                 # 禁用wandb等日志上报
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,   
)

# 6. 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 8. 开始训练
trainer.train()

# 9. 保存模型
model.save_pretrained(f"{output_dir}/lora_adapter")
tokenizer.save_pretrained(f"{output_dir}/lora_adapter")

if __name__ == "__main__":
 main()
```

deploy_lora_model.py

模型部署使用

方法一，使用 PEFT (Hugging Face)

这是最通用、最基础的方法。先加载基础模型，再通过PEF（Parameter-Efficient Fine-Tuning，参数高效微调）库把适配器挂载上去[](https://blog.csdn.net/weixin_42509513/article/details/157073871)。

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

txt = """保持原段落的写作风格，对其进行扩展和深化。
Paragraph:
###
当白鹿仓的总乡约田福贤要鹿子霖出任第一保障所的乡约那阵儿，鹿子霖听着别扭的「保障所」和别扭的「乡约」这些新名称满腹狐疑，拿不定主意，推委说自己要做庄稼，怕没时间办保障所里的事。当他从县府接受训练回来以后，就对田福贤是一种知遇恩情的感激心情了
鹿子霖在县府接受了为期半月的任职训练。受训结束的前一天，县长史维华再一次到场训示，发给大家每人一身青色制服，换上了一色一式制服的各仓总乡约和各保障所的乡约们一起同史县长合影留念，这无疑是滋水县历史上别开生面的一张历史性照片。鹿子霖脱下长袍马褂，穿上新制服到大镜前一照，自己先吓了一跳，几乎认不出自己了。停了片刻，他还是相信那个穿一身青色洋布制服的鹿子霖，仍是那个穿长袍马褂的鹿子霖：长条脸，高额头，深陷的眼睛，长长的眼睫毛，统直的鼻子，俊俏的嘴角，这个鹿子霖比那个鹿子霖更显得精神了
###
一天后晌，两个正在朱先生的白鹿书院念书的儿子闻讯跑到县府来看望他，看见他一身制服就惊得愣呆呆地瞅着。"""
# 1. 加载基础模型
base_model_path = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0"   # 或 "cuda" (单卡))
)

# 2. 加载LoRA适配器
lora_adapter_path = f"/content/lora_adapter"
model = PeftModel.from_pretrained(model, lora_adapter_path)

# 3. 合并并设置为推理模式 (可选但推荐)
model = model.merge_and_unload()
model.eval()

# 4. 进行推理
inputs = tokenizer(txt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

方法二，使用 vLLM (高性能批量推理)

vLLM是为高吞吐量设计的推理框架，支持在离线推理中动态加载LoRA[](https://docs.vllm.ai/en/v0.10.1.1/features/lora.html)[](https://docs.vllm.com.cn/en/latest/features/lora/)

```
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

txt = """保持原段落的写作风格，对其进行扩展和深化。
Paragraph:
###
当白鹿仓的总乡约田福贤要鹿子霖出任第一保障所的乡约那阵儿，鹿子霖听着别扭的「保障所」和别扭的「乡约」这些新名称满腹狐疑，拿不定主意，推委说自己要做庄稼，怕没时间办保障所里的事。当他从县府接受训练回来以后，就对田福贤是一种知遇恩情的感激心情了
鹿子霖在县府接受了为期半月的任职训练。受训结束的前一天，县长史维华再一次到场训示，发给大家每人一身青色制服，换上了一色一式制服的各仓总乡约和各保障所的乡约们一起同史县长合影留念，这无疑是滋水县历史上别开生面的一张历史性照片。鹿子霖脱下长袍马褂，穿上新制服到大镜前一照，自己先吓了一跳，几乎认不出自己了。停了片刻，他还是相信那个穿一身青色洋布制服的鹿子霖，仍是那个穿长袍马褂的鹿子霖：长条脸，高额头，深陷的眼睛，长长的眼睫毛，统直的鼻子，俊俏的嘴角，这个鹿子霖比那个鹿子霖更显得精神了
###
一天后晌，两个正在朱先生的白鹿书院念书的儿子闻讯跑到县府来看望他，看见他一身制服就惊得愣呆呆地瞅着。"""

# 1. 初始化基础模型，必须开启LoRA支持
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    enable_lora=True,           # 关键参数
    max_loras=1,                # 最大同时加载的LoRA数量
    max_lora_rank=64,           # 根据适配器rank设置
)

# 2. 定义LoRA请求
lora_request = LoRARequest(
    lora_name="my_lora_adapter",                # 自定义名称
    lora_int_id=1,                              # 唯一数字ID
    lora_path="/content/lora_adapter"      # 适配器路径
)

# 3. 准备提示词和采样参数
prompts = [txt]
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

# 4. 执行推理，传入lora_request
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

for output in outputs:
    print(output.outputs[0].text)
```
