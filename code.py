!pip install transformers








from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
#  Load Model and Tokenize
tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")
# Encode Input Text
input_text = ''
input_ids = tokenizer.encode(input_text, return_tensors="pt")
# Generate Summary Text Ids
summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=2.0,
    max_length=142, #요악문의 최대 길이 설정
    min_length=5,   #요약문의 최소 길이 설정
    num_beams=4,    #문장 생성시 다음 단어를  탐색하는 영역의 개수
)
# Decoding Text
print(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))
