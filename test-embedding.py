from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
# from fast_langdetect import detect_language
from google.transliteration import transliterate_text
from aksharamukha import transliterate
import string
from indictrans import Transliterator
from stuff_to_be_imported import *

def is_hinglish(text, threshold=0.6):
    # Fast pre-processing
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    if not words: return False

    english_count = sum(1 for w in words if w in english_stopwords)
    hinglish_count = sum(1 for w in words if w in hinglish_stopwords)
    total_count = english_count + hinglish_count

    # Explicit Hinglish term check if no stopwords matched
    if total_count == 0:
        return any(w in hinglish_stopwords for w in words)

    return (hinglish_count / total_count) >= threshold

# ---------------------- SentenceTransformer Pipeline ----------------------

class EmbeddingPipeline:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, sentence):
        return np.array(self.model.encode([sentence]))[0]
    
    def compare_sentences(self, sentence1, sentence2):
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity

# ---------------------- IndicBERT (HuggingFace) Pipeline ----------------------

class IndicBERTPipeline:
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-only"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def get_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            embedding = last_hidden_state.mean(dim=1).squeeze()  # [hidden_dim]
        return embedding.numpy()

    def compare_sentences(self, sentence1, sentence2):
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity
    

# ---------------------- Translation + Embedding Pipeline ----------------------
class HinglishTranslationEmbeddingPipeline:
    def __init__(self, translation_model="rudrashah/RLM-hinglish-translator", embed_model="sentence-transformers/all-MiniLM-L6-v2",
                 source_lang="Hinglish", target_lang="English"):
        self.model_name = f"{source_lang}2{target_lang}+{embed_model}"
        self.translator = pipeline("text2text-generation", model=translation_model)
        self.embedder = SentenceTransformer(embed_model)
        self.source_lang = source_lang
        self.target_lang = target_lang

    
    def translate(self, sentence):
        template = f"{self.source_lang}:\n{sentence}\n\n{self.target_lang}:\n"
        result = self.translator(template)[0]['generated_text']

        if f"\n\n{self.target_lang}:\n" in result:
            result = result.split(f"\n\n{self.target_lang}:\n")[1].strip()
        return result

    def get_embedding(self, sentence):
        translated_text = self.translate(sentence)
        return np.array(self.embedder.encode([translated_text]))[0]
    
    def compare_sentences(self, sentence1, sentence2):
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity

# ---------------------- Transliteration + Embedding Pipeline ----------------------
class TransliterationEmbeddingPipeline:
    def __init__(self, embed_model="krutrim-ai-labs/vyakyarth", transliteration_model="indic-trans"):
        self.model_name = f"{embed_model}+{transliteration_model}"
        self.embedder = SentenceTransformer(embed_model)
        self.trn = Transliterator(source='eng', target='hin', build_lookup=True)
        self.transliteration_model = transliteration_model
    
    def transliterate(self, sentence):
        try:
            if self.transliteration_model == "indic-trans":
                result = self.trn.transform(sentence)
            elif self.transliteration_model == "aksharamukha":
                result = transliterate.process(src='IAST', tgt='Devanagari', txt=sentence)
            elif self.transliteration_model == "google":
                result = transliterate_text(sentence, lang_code='hi')

            return result
        except Exception as e:
            print(f"Transliteration failed: {e}")
            return sentence

    def get_embedding(self, sentence):
        if not is_hinglish(sentence):
            transliterated_text = sentence
        else:
            transliterated_text = self.transliterate(sentence)
            print(f"Transliterating Hinglish sentence: {sentence}, Result: {transliterated_text}")

        return np.array(self.embedder.encode([transliterated_text]))[0]
    
    def compare_sentences(self, sentence1, sentence2):
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity


# ---------------------- Comparison Function ----------------------

def compare_with_all_pipelines(pipelines, sentence1, sentence2):
    results = {}
    for pipeline in pipelines:
        similarity = pipeline.compare_sentences(sentence1, sentence2)
        results[pipeline.model_name] = similarity
    return results

# ---------------------- Main Application ----------------------

def main():
    # Initialize all pipelines (your SentenceTransformer + custom IndicBERT pipeline)
    pipelines = [
        # EmbeddingPipeline("krutrim-ai-labs/vyakyarth"),
        # EmbeddingPipeline("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        # HinglishTranslationEmbeddingPipeline(),
        TransliterationEmbeddingPipeline(transliteration_model="indic-trans"),
        # TransliterationEmbeddingPipeline(transliteration_model="aksharamukha"),
        # TransliterationEmbeddingPipeline(transliteration_model="google"),
    ]
    
    print("Unified Sentence Embedding Comparison Tool")
    print("Enter 'q' at any time to quit")
    
    while True:
        print("\n1 >>Compare two sentences")
        print("2 3 4 >> Test with high mid low examples")
        print("q. Quit")
        
        choice = input("Choose an option: ")
        
        if choice.lower() == 'q':
            break
            
        elif choice == '1':
            sentence1 = input("Enter first sentence: ")
            if sentence1.lower() == 'q':
                break
                
            sentence2 = input("Enter second sentence: ")
            if sentence2.lower() == 'q':
                break
                
            results = compare_with_all_pipelines(pipelines, sentence1, sentence2)
            print("\nSimilarity scores across models:")
            for model_name, similarity in results.items():
                print(f"- {model_name}: {similarity:.6f}")
            
        elif choice in ['2', '3', '4']:
            if choice == '2':
                print("\nTesting with High Similarity Examples")
                examples = high_examples
            elif choice == '3':
                print("\nTesting with Mid Similarity Examples")
                examples = moderate_examples
            elif choice == '4':
                print("\nTesting with Low Similarity Examples")
                examples = low_examples

            similarity_avg = [0]*len(pipelines)
            
            for i, (s1, s2) in enumerate(examples):
                print(f"\nExample {i+1}: '{s1}' vs '{s2}'")
                results = compare_with_all_pipelines(pipelines, s1, s2)
                print("Similarity scores across models:")
                for model_name, similarity in results.items():
                    print(f"- {model_name}: {similarity:.6f}")
                
                for j, pipeline in enumerate(pipelines):
                    similarity_avg[j] += results[pipeline.model_name]
            # Average the similarity scores
            num_examples = len(examples)
            print("\nAverage Similarity Scores:")
            for j, pipeline in enumerate(pipelines):
                avg_score = similarity_avg[j] / num_examples
                print(f"- {pipeline.model_name}: {avg_score:.6f}")
                print(f"Total examples: {num_examples}")
            print("\nTesting complete!")
                    


def test_translation():
    print("\nTesting Hinglish to English Translation")
    print("======================================")
    
    # Initialize the translation pipeline
    translator = HinglishTranslationEmbeddingPipeline()
    
    # Test examples
    examples = [
        "main kal movie dekhne jaa raha hoon",
        "kya aap free ho aaj sham ko coffee ke liye?",
        "mujhe yeh product bahut acha laga",
        "homework complete karna hai weekend tak",
        "train 2 ghante late hai because of rain"
    ]
    
    for i, text in enumerate(examples):
        translated = translator.translate(text)
        print(f"\nExample {i+1}:")
        print(f"Original: {text}")
        print(f"Translated: {translated}")
    
    # Interactive testing
    print("\nEnter 'q' to quit testing translation")
    while True:
        user_input = input("\nEnter Hinglish text to translate: ")
        if user_input.lower() == 'q':
            break
        
        translated = translator.translate(user_input)
        print(f"Translated: {translated}")



if __name__ == "__main__":
    main()
    # test_translation()


