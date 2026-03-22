from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "ByteDance/Ouro-2.6B-Thinking"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

print("Model loaded successfully!")
print(f"Model is on: {next(model.parameters()).device}")

model.config.early_exit_threshold = 0.87

while True:
    prompt = input("\nEnter prompt (or 'quit' to exit): ")
    if prompt.lower() == "quit":
        break

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(inputs)

    print("Generating...")
    outputs = model.generate(
        inputs,attention_mask=attention_mask,
        max_new_tokens=1024,
        temperature=1.0,
        top_p=0.7
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
