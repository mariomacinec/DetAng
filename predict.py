import json
import os
import torch
from transformers import AutoTokenizer, BertForTokenClassification
import config

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
    model = BertForTokenClassification.from_pretrained(config.MODEL_SAVE_PATH)
    model.eval()
    return model, tokenizer

def predict_sentences(sentences, model, tokenizer):
    results = []
    # Gets list of strings, returns a dictionary
    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue

        # Entry tokenization
        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LEN
        )

        # Inference (calculation of predictions)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: [1, seq_len, num_labels]
            predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()

        # Mining tags back to words
        word_ids = inputs.word_ids(batch_index=0)
        detected_anglicisms = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            # Ignoring [CLS], [SEP], [PAD] and another subwords of the same word
            if word_idx is not None and word_idx != previous_word_idx:
                pred_label = predictions[idx]
                if pred_label == 1:
                    detected_anglicisms.append(words[word_idx])
                previous_word_idx = word_idx

        # Connecting the result for a sentence
        has_anglicism = len(detected_anglicisms) > 0

        results.append({
            "sentence": sentence.strip(),
            "has_anglicism": has_anglicism,
            "detected_words": list(set(detected_anglicisms)) # Unique found words
        })

    return results

def predict_from_file(file_path):
    # Loads sentences from a text file
    model, tokenizer = load_model_and_tokenizer()

    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()

    predictions = predict_sentences(sentences, model, tokenizer)
    return predictions

if __name__ == "__main__":
    test_file = "data/predict_text.txt"
    output_json_path = "data/latest_predict_results.json"

    results = predict_from_file(test_file)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nPredictions")
    for res in results:
        status = "Has anglicism." if res["has_anglicism"] else "No anglicism."
        print(f"\nSentence: {res['sentence']}")
        print(f"Result: {status}")
        if res["has_anglicism"]:
            print(f"Detected words: {', '.join(res['detected_words'])}")

    print(f"\nResults saved to: {output_json_path}")
