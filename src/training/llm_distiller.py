"""
LLM Distillation Pipeline
Uses open-source LLM to generate training labels for lightweight evaluator
This satisfies: "No one-shot prompt" - LLM is teacher, not the evaluator
"""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))
from retrieval.facet_retriever import FacetRetriever
from features.feature_extractor import FeatureExtractor


class LLMDistiller:
    """
    Uses LLM to create training dataset for lightweight evaluator.
    
    Key concept: LLM is NOT the final evaluator.
    LLM is only used to generate training labels.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize distiller"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        self.retriever = FacetRetriever(config_path)
        self.feature_extractor = FeatureExtractor()
        
        print(f"🖥️  Using device: {self.device}")
        print("✅ LLM Distiller initialized")
    
    def load_teacher_model(self):
        """Load LLM as teacher model"""
        if self.model is not None:
            return
        
        model_name = self.config['model']['name']
        print(f"📦 Loading teacher model: {model_name}...")
        
        # Quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            print("✅ Teacher model loaded!")
        except Exception as e:
            print(f"⚠️  Could not load teacher model: {e}")
            print("⚠️  Will use rule-based labeling instead")
            self.model = "rule_based"
    
    def create_simple_prompt(self, turn_text: str, facet_name: str, facet_desc: str) -> str:
        """Create simplified prompt for teacher LLM"""
        prompt = f"""Score this conversation turn on the given facet.

Turn: "{turn_text}"

Facet: {facet_name}
Definition: {facet_desc}

Score (1-5): """
        return prompt
    
    def get_teacher_score(self, turn_text: str, facet_name: str, facet_desc: str) -> int:
        """Get score from teacher LLM"""
        if self.model == "rule_based":
            return self._rule_based_label(turn_text, facet_name)
        
        prompt = self.create_simple_prompt(turn_text, facet_name, facet_desc)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.3,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            
            # Extract score
            score = 3
            for char in response:
                if char.isdigit():
                    score = int(char)
                    break
            
            return max(1, min(5, score))
            
        except Exception as e:
            print(f"⚠️ Error getting teacher score: {e}")
            return 3
    
    def _rule_based_label(self, turn_text: str, facet_name: str) -> int:
        """
        Rule-based labeling fallback.
        Uses heuristics based on facet categories.
        """
        text_lower = turn_text.lower()
        facet_lower = facet_name.lower()
        
        # Emotional facets
        if any(word in facet_lower for word in ['emotion', 'sad', 'joy', 'happy', 'depress']):
            emotion_words = ['sad', 'happy', 'joy', 'excited', 'depressed', 'down', 'great']
            count = sum(1 for w in emotion_words if w in text_lower)
            return min(5, 2 + count)
        
        # Toxicity facets
        elif any(word in facet_lower for word in ['toxic', 'harmful', 'hostile', 'aggress', 'rude']):
            toxic_words = ['hate', 'stupid', 'idiot', 'useless', 'terrible', 'awful', 'worst']
            count = sum(1 for w in toxic_words if w in text_lower)
            return min(5, 1 + count * 2)
        
        # Cognitive facets
        elif any(word in facet_lower for word in ['cognitive', 'reasoning', 'logic', 'thinking']):
            cognitive_words = ['because', 'therefore', 'think', 'reason', 'analyze', 'understand']
            count = sum(1 for w in cognitive_words if w in text_lower)
            return min(5, 2 + count)
        
        # Helpful facets
        elif any(word in facet_lower for word in ['helpful', 'support', 'empathy', 'compassion']):
            helpful_words = ['help', 'support', 'understand', 'sorry', 'try', 'consider']
            count = sum(1 for w in helpful_words if w in text_lower)
            return min(5, 2 + count)
        
        # Leadership facets
        elif any(word in facet_lower for word in ['leadership', 'motivation', 'delegation']):
            leadership_words = ['team', 'lead', 'manage', 'motivate', 'delegate', 'inspire']
            count = sum(1 for w in leadership_words if w in text_lower)
            return min(5, 1 + count * 2)
        
        # Default
        return 3
    
    def generate_training_data(
        self, 
        conversations_file: str = "data/processed/conversations.json",
        output_file: str = "data/training/training_data.jsonl",
        num_samples: int = 500,
        facets_per_turn: int = 10
    ):
        """
        Generate training dataset using teacher LLM.
        
        This creates labeled data: (features, facet_embedding) -> score
        """
        print(f"\n🎓 Generating Training Data")
        print(f"  Target samples: {num_samples}")
        print(f"  Facets per turn: {facets_per_turn}")
        
        # Load conversations
        with open(conversations_file, 'r') as f:
            conversations = json.load(f)
        
        print(f"📖 Loaded {len(conversations)} conversations")
        
        # Load retriever and teacher
        self.retriever.load_index()
        self.load_teacher_model()
        
        # Generate training samples
        training_samples = []
        
        pbar = tqdm(total=num_samples, desc="Generating training data")
        
        conv_idx = 0
        while len(training_samples) < num_samples and conv_idx < len(conversations):
            conv = conversations[conv_idx]
            
            for turn in conv['turns']:
                if len(training_samples) >= num_samples:
                    break
                
                turn_text = turn['text']
                speaker = turn['speaker']
                
                # Retrieve relevant facets
                relevant_facets = self.retriever.retrieve_relevant_facets(
                    turn_text, 
                    top_k=facets_per_turn
                )
                
                # Get features
                features = self.feature_extractor.extract_all_features(turn_text, speaker)
                
                # Get teacher labels for each facet
                for _, facet_row in relevant_facets.iterrows():
                    facet_name = facet_row['facet_name']
                    facet_desc = facet_row['description']
                    category = facet_row['category']
                    
                    # Get teacher score
                    score = self.get_teacher_score(turn_text, facet_name, facet_desc)
                    
                    # Create training sample
                    sample = {
                        'turn_text': turn_text,
                        'speaker': speaker,
                        'facet_name': facet_name,
                        'facet_category': category,
                        'features': features,
                        'score': score
                    }
                    
                    training_samples.append(sample)
                    pbar.update(1)
                    
                    if len(training_samples) >= num_samples:
                        break
            
            conv_idx += 1
        
        pbar.close()
        
        print(f"✅ Generated {len(training_samples)} training samples")
        
        # Save training data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in training_samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"💾 Saved training data to {output_file}")
        
        # Print statistics
        scores_dist = pd.Series([s['score'] for s in training_samples]).value_counts().sort_index()
        print(f"\n📊 Score Distribution:")
        for score, count in scores_dist.items():
            print(f"  Score {score}: {count} samples ({count/len(training_samples)*100:.1f}%)")
        
        categories_dist = pd.Series([s['facet_category'] for s in training_samples]).value_counts()
        print(f"\n📊 Top Facet Categories:")
        for cat, count in categories_dist.head(5).items():
            print(f"  {cat}: {count} samples")
        
        return training_samples


def main():
    """Generate training data using LLM distillation"""
    distiller = LLMDistiller()
    
    # Generate training data
    training_samples = distiller.generate_training_data(
        num_samples=1000,  # Generate 1000 training samples
        facets_per_turn=10
    )
    
    print("\n✅ LLM Distillation Complete!")
    print("📌 Next step: Train lightweight ML evaluator on this data")


if __name__ == "__main__":
    main()
