from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_bart_tl_topics(input, model, tokenizer):
    enc = tokenizer(input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=15,
        min_length=1,
        do_sample=False,
        num_beams=25,
        length_penalty=1.0,
        repetition_penalty=1.5
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded

if __name__ == "__main__":
    input = "site web google search website online internet social content user"
    
    mname = "cristian-popa/bart-tl-ng"
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)

    topics = get_bart_tl_topics(input, model, tokenizer)
    print(f"input: {input}")
    print(f"topics: {topics}")