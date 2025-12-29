"""
Feature Extraction for Conversation Turns
Extracts linguistic, emotional, and behavioral features for ML evaluation
"""

import numpy as np
from typing import Dict, List
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re


class FeatureExtractor:
    """
    Extract features from conversation turns for ML-based evaluation.
    These features replace prompt-based scoring.
    """
    
    def __init__(self):
        """Initialize feature extractors"""
        self.vader = SentimentIntensityAnalyzer()
        print("✅ Feature extractor initialized")
    
    def extract_basic_features(self, text: str) -> Dict:
        """Extract basic linguistic features"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': max(1, len([s for s in sentences if s.strip()])),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)),
        }
    
    def extract_sentiment_features(self, text: str) -> Dict:
        """Extract sentiment and emotion features"""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        
        return {
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
        }
    
    def extract_emotion_features(self, text: str) -> Dict:
        """Extract emotion-related features using keyword matching"""
        text_lower = text.lower()
        
        # Emotion keywords
        joy_words = ['happy', 'joy', 'excited', 'glad', 'wonderful', 'great', 'amazing', 'love', 'excellent']
        sadness_words = ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'sorry', 'hurt', 'pain']
        anger_words = ['angry', 'mad', 'furious', 'hate', 'annoyed', 'irritated', 'frustrated']
        fear_words = ['afraid', 'scared', 'fear', 'anxious', 'worried', 'nervous', 'panic']
        surprise_words = ['surprise', 'shocked', 'amazed', 'astonished', 'unexpected']
        disgust_words = ['disgusting', 'gross', 'awful', 'terrible', 'horrible']
        
        return {
            'emotion_joy': sum(1 for w in joy_words if w in text_lower),
            'emotion_sadness': sum(1 for w in sadness_words if w in text_lower),
            'emotion_anger': sum(1 for w in anger_words if w in text_lower),
            'emotion_fear': sum(1 for w in fear_words if w in text_lower),
            'emotion_surprise': sum(1 for w in surprise_words if w in text_lower),
            'emotion_disgust': sum(1 for w in disgust_words if w in text_lower),
        }
    
    def extract_toxicity_features(self, text: str) -> Dict:
        """Extract toxicity and safety features"""
        text_lower = text.lower()
        
        # Toxic keywords
        toxic_words = ['hate', 'stupid', 'idiot', 'useless', 'terrible', 'awful', 'worst', 'kill', 'die']
        profanity = ['damn', 'hell', 'crap', 'suck']
        
        return {
            'toxicity_score': sum(1 for w in toxic_words if w in text_lower),
            'profanity_count': sum(1 for w in profanity if w in text_lower),
            'has_capslock': int(any(word.isupper() and len(word) > 2 for word in text.split())),
        }
    
    def extract_helpfulness_features(self, text: str) -> Dict:
        """Extract features related to helpfulness and support"""
        text_lower = text.lower()
        
        # Helpful keywords
        helpful_words = ['help', 'support', 'assist', 'suggest', 'recommend', 'try', 'consider', 'maybe']
        empathy_words = ['understand', 'feel', 'sorry', 'care', 'empathize', 'know how']
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        
        return {
            'helpfulness_score': sum(1 for w in helpful_words if w in text_lower),
            'empathy_score': sum(1 for w in empathy_words if w in text_lower),
            'asks_questions': sum(1 for w in question_words if w in text_lower),
            'offers_suggestion': int('try' in text_lower or 'suggest' in text_lower or 'consider' in text_lower),
        }
    
    def extract_cognitive_features(self, text: str) -> Dict:
        """Extract cognitive and reasoning features"""
        text_lower = text.lower()
        
        # Cognitive keywords
        reasoning_words = ['because', 'therefore', 'thus', 'hence', 'since', 'so', 'reason']
        cognitive_words = ['think', 'analyze', 'consider', 'understand', 'know', 'realize', 'believe']
        complexity_words = ['complex', 'complicated', 'difficult', 'intricate']
        
        return {
            'reasoning_indicators': sum(1 for w in reasoning_words if w in text_lower),
            'cognitive_words': sum(1 for w in cognitive_words if w in text_lower),
            'complexity_level': sum(1 for w in complexity_words if w in text_lower),
            'has_numbers': int(bool(re.search(r'\d', text))),
        }
    
    def extract_conversational_features(self, text: str, speaker: str) -> Dict:
        """Extract conversational style features"""
        text_lower = text.lower()
        
        # Conversational markers
        first_person = ['i', 'me', 'my', 'mine', 'myself']
        second_person = ['you', 'your', 'yours', 'yourself']
        
        return {
            'is_user': int(speaker.lower() == 'user'),
            'is_ai': int(speaker.lower() == 'ai'),
            'first_person_count': sum(1 for w in first_person if f' {w} ' in f' {text_lower} '),
            'second_person_count': sum(1 for w in second_person if f' {w} ' in f' {text_lower} '),
            'starts_with_question': int(text.strip().startswith(('what', 'why', 'how', 'when', 'where', 'who'))),
            'politeness_markers': sum(1 for w in ['please', 'thank', 'sorry', 'excuse'] if w in text_lower),
        }
    
    def extract_all_features(self, text: str, speaker: str = "user", context: str = "") -> Dict:
        """
        Extract all features for a conversation turn.
        
        Returns a feature dictionary that will be used by the ML evaluator.
        """
        features = {}
        
        # Combine all feature sets
        features.update(self.extract_basic_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_emotion_features(text))
        features.update(self.extract_toxicity_features(text))
        features.update(self.extract_helpfulness_features(text))
        features.update(self.extract_cognitive_features(text))
        features.update(self.extract_conversational_features(text, speaker))
        
        # Context features
        if context:
            features['context_length'] = len(context.split())
        else:
            features['context_length'] = 0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Extract from a dummy text to get all feature keys
        dummy_features = self.extract_all_features("dummy text", "user")
        return list(dummy_features.keys())


def main():
    """Test feature extraction"""
    print("🧪 Testing Feature Extraction\n")
    
    extractor = FeatureExtractor()
    
    test_cases = [
        ("I've been feeling really down and nothing makes me happy.", "User", "emotional"),
        ("You're completely useless! This is terrible advice!", "User", "toxic"),
        ("Let me explain the solution step by step using logical reasoning.", "AI", "cognitive"),
        ("I understand how you feel. Have you considered talking to someone?", "AI", "helpful"),
    ]
    
    for text, speaker, category in test_cases:
        print(f"{'='*60}")
        print(f"Text: {text}")
        print(f"Speaker: {speaker}")
        print(f"Expected: {category}")
        print(f"{'-'*60}")
        
        features = extractor.extract_all_features(text, speaker)
        
        # Print key features
        print("\n🔑 Key Features:")
        key_features = [
            'vader_compound', 'emotion_sadness', 'emotion_joy', 'emotion_anger',
            'toxicity_score', 'helpfulness_score', 'empathy_score', 'reasoning_indicators'
        ]
        
        for feat in key_features:
            if feat in features:
                print(f"  {feat}: {features[feat]:.3f}" if isinstance(features[feat], float) else f"  {feat}: {features[feat]}")
        
        print()
    
    print(f"{'='*60}")
    print(f"\n✅ Total features extracted: {len(extractor.get_feature_names())}")
    print(f"📋 Feature names: {extractor.get_feature_names()[:10]}... (showing first 10)")
    print("\n✅ Feature extraction ready!")


if __name__ == "__main__":
    main()
