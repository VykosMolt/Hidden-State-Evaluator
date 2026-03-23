from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "ByteDance/Ouro-2.6B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

captured = {}

def hook_fn(module, input, output):
    # output is (outputs, hidden_states_list, gate_list)
    captured['hidden_states_list'] = output[1]
    captured['gate_list'] = output[2]

model.model.register_forward_hook(hook_fn)

messages = [{"role": "user", "content": "What is 2+2?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

attention_mask = torch.ones_like(inputs)

with torch.no_grad():
    outputs = model(
        inputs,
        attention_mask=attention_mask,
        return_dict=True
    )

print("Number of loop steps:", len(captured['hidden_states_list']))
print("Hidden state shape per step:", captured['hidden_states_list'][0].shape)
print("Number of gate values:", len(captured['gate_list']))
print("Gate values (mean per step):", [g.mean().item() for g in captured['gate_list']])
