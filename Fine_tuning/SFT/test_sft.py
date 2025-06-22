from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration

model_name1 = "KhushalM/Qwen2-VL-2B-Instruct-SFT"
model_name2 = "Qwen/Qwen2-VL-2B-Instruct"

tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
model1 = Qwen2VLForConditionalGeneration.from_pretrained(model_name1)

tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = Qwen2VLForConditionalGeneration.from_pretrained(model_name2)

sample_questions = ["Explain Reinforcement Learning", "The article on my screen is about Domestic Debt, explain it so that it is easy to understand"]

for question in sample_questions:
    inputs = tokenizer1.apply_chat_template([{"role": "user", "content": question}], tokenize=False)
    outputs = model1.generate(inputs, max_new_tokens=1000)
    print(tokenizer1.decode(outputs[0], skip_special_tokens=True))

for question in sample_questions:
    inputs = tokenizer2.apply_chat_template([{"role": "user", "content": question}], tokenize=False)
    outputs = model2.generate(inputs, max_new_tokens=1000)
    print(tokenizer2.decode(outputs[0], skip_special_tokens=True))
    print("-"*100)




