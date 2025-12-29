"""
Facet Cleaner and Categorizer
Processes the raw Facets_Assignment.csv and creates a structured facet library
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List

class FacetCleaner:
    """Clean and categorize facets from the raw dataset"""
    
    def __init__(self, input_file: str = "data/raw/Facets_Assignment.csv"):
        self.input_file = input_file
        self.facet_categories = self._define_categories()
        
    def _define_categories(self) -> Dict[str, List[str]]:
        """Define keyword-based categorization rules"""
        return {
            "Emotional": [
                "emotion", "feeling", "affect", "mood", "joy", "sad", "happy", 
                "depression", "anxiety", "stress", "empathy", "compassion", 
                "anger", "fear", "disgust", "surprise", "trust", "anticipation",
                "bliss", "contentment", "distress", "burnout", "fatigue"
            ],
            "Personality": [
                "personality", "trait", "character", "honesty", "humility", 
                "openness", "conscientiousness", "extraversion", "agreeableness",
                "neuroticism", "hexaco", "big five", "assertiveness", "boldness",
                "introversion", "shyness", "confidence", "self-esteem"
            ],
            "Cognitive": [
                "iq", "intelligence", "reasoning", "memory", "attention", 
                "cognitive", "thinking", "problem-solving", "analytical",
                "logical", "working memory", "spatial", "verbal", "numerical",
                "creativity", "innovation", "comprehension", "understanding"
            ],
            "Behavioral": [
                "behavior", "action", "habit", "activity", "initiative",
                "collaboration", "cooperation", "participation", "engagement",
                "perseverance", "persistence", "risk-taking", "decision-making",
                "impulsivity", "planning", "organizing"
            ],
            "Social": [
                "social", "relationship", "communication", "interpersonal",
                "listening", "speaking", "eye contact", "non-verbal", "empathy",
                "trust", "friendship", "networking", "interaction", "charisma",
                "influence", "persuasion"
            ],
            "Leadership": [
                "leadership", "management", "delegation", "motivating",
                "inspiring", "directing", "democratic", "autocratic",
                "transformational", "transactional", "team-building"
            ],
            "Moral_Ethical": [
                "moral", "ethical", "integrity", "honesty", "justice",
                "fairness", "dignity", "respect", "civility", "virtue",
                "righteousness", "principle"
            ],
            "Safety_Toxicity": [
                "safety", "toxicity", "harmful", "dangerous", "aggressive",
                "hostile", "violent", "harassment", "bullying", "discrimination",
                "hate", "disrespect"
            ],
            "Motivation": [
                "motivation", "drive", "goal", "achievement", "ambition",
                "aspiration", "desire", "incentive", "intrinsic", "extrinsic"
            ],
            "Health_Biological": [
                "health", "biological", "physiological", "hormone", "sleep",
                "diet", "nutrition", "exercise", "fitness", "medical",
                "symptom", "disorder", "chronic", "pain"
            ],
            "Spiritual_Religious": [
                "spiritual", "religious", "meditation", "prayer", "pilgrimage",
                "faith", "belief", "sacred", "divine", "ritual", "practice",
                "yoga", "mindfulness", "buddhist", "hindu", "islamic", "christian"
            ],
            "Work_Professional": [
                "work", "job", "career", "professional", "deadline", "productivity",
                "efficiency", "performance", "skill", "expertise", "training",
                "meeting", "project"
            ],
            "Linguistic": [
                "language", "grammar", "vocabulary", "syntax", "spelling",
                "writing", "reading", "verbal", "linguistic", "communication",
                "clarity", "coherence", "fluency"
            ]
        }
    
    def clean_facet_name(self, facet: str) -> str:
        """Clean individual facet name"""
        if pd.isna(facet):
            return None
            
        # Convert to string and strip
        facet = str(facet).strip()
        
        # Remove numbering (e.g., "800. " or "754.")
        facet = re.sub(r'^\d+\.\s*', '', facet)
        
        # Remove trailing colons
        facet = facet.rstrip(':')
        
        # Normalize spaces
        facet = re.sub(r'\s+', ' ', facet)
        
        # Skip if empty after cleaning
        if not facet or len(facet) < 2:
            return None
            
        return facet
    
    def categorize_facet(self, facet_name: str) -> str:
        """Categorize facet based on keywords"""
        facet_lower = facet_name.lower()
        
        # Check each category
        for category, keywords in self.facet_categories.items():
            for keyword in keywords:
                if keyword in facet_lower:
                    return category
        
        # Default category
        return "Other"
    
    def generate_description(self, facet_name: str, category: str) -> str:
        """Generate a description for the facet"""
        descriptions = {
            "Emotional": f"Measures emotional state or expression related to {facet_name}",
            "Personality": f"Evaluates personality trait: {facet_name}",
            "Cognitive": f"Assesses cognitive ability or mental process: {facet_name}",
            "Behavioral": f"Evaluates behavioral pattern: {facet_name}",
            "Social": f"Measures social skill or interaction quality: {facet_name}",
            "Leadership": f"Assesses leadership quality: {facet_name}",
            "Moral_Ethical": f"Evaluates moral or ethical dimension: {facet_name}",
            "Safety_Toxicity": f"Detects safety concern or toxic behavior: {facet_name}",
            "Motivation": f"Measures motivational aspect: {facet_name}",
            "Health_Biological": f"Health or biological metric: {facet_name}",
            "Spiritual_Religious": f"Spiritual or religious practice: {facet_name}",
            "Work_Professional": f"Work-related competency: {facet_name}",
            "Linguistic": f"Linguistic quality measure: {facet_name}",
            "Other": f"Evaluates: {facet_name}"
        }
        return descriptions.get(category, f"Evaluates: {facet_name}")
    
    def should_keep_facet(self, facet_name: str) -> bool:
        """Determine if facet should be kept (filter out very niche biological/mystical ones)"""
        # Skip extremely specific biological markers
        skip_keywords = [
            "hexagram", "aura color", "astrology rising", "archon meditation",
            "fsh level", "chromatin-accessibility", "serotonin transporter",
            "parathyroid-hormone", "basophil count"
        ]
        
        facet_lower = facet_name.lower()
        for skip_word in skip_keywords:
            if skip_word in facet_lower:
                return False
        
        return True
    
    def process_facets(self) -> pd.DataFrame:
        """Main processing pipeline"""
        print("📂 Loading raw facets...")
        
        # Read the CSV (assuming single column with facet names)
        df = pd.read_csv(self.input_file, header=None, names=['raw_facet'])
        
        print(f"📊 Found {len(df)} raw facets")
        
        # Clean facet names
        print("🧹 Cleaning facet names...")
        df['facet_name'] = df['raw_facet'].apply(self.clean_facet_name)
        
        # Remove nulls
        df = df.dropna(subset=['facet_name'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['facet_name'])
        
        # Filter out unwanted facets
        df = df[df['facet_name'].apply(self.should_keep_facet)]
        
        print(f"✅ {len(df)} facets after cleaning")
        
        # Categorize
        print("🏷️  Categorizing facets...")
        df['category'] = df['facet_name'].apply(self.categorize_facet)
        
        # Generate descriptions
        print("📝 Generating descriptions...")
        df['description'] = df.apply(
            lambda row: self.generate_description(row['facet_name'], row['category']),
            axis=1
        )
        
        # Add metadata
        df['facet_id'] = range(1, len(df) + 1)
        df['scale_min'] = 1
        df['scale_max'] = 5
        df['scale_type'] = 'ordinal'
        
        # Reorder columns
        df = df[[
            'facet_id', 'facet_name', 'category', 'description',
            'scale_min', 'scale_max', 'scale_type'
        ]]
        
        # Sort by category and name
        df = df.sort_values(['category', 'facet_name'])
        df = df.reset_index(drop=True)
        
        return df
    
    def save_processed_facets(self, df: pd.DataFrame, output_file: str = "data/processed/facets_structured.csv"):
        """Save processed facets"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {len(df)} structured facets to {output_file}")
        
        # Print summary
        print("\n📊 Category Distribution:")
        print(df['category'].value_counts())
    
    def run(self):
        """Execute full pipeline"""
        df = self.process_facets()
        self.save_processed_facets(df)
        return df


if __name__ == "__main__":
    # First, let's copy the raw facets data
    print("🚀 Starting Facet Processing Pipeline\n")
    
    cleaner = FacetCleaner()
    structured_facets = cleaner.run()
    
    print("\n✅ Facet processing complete!")
    print(f"📈 Total facets ready for evaluation: {len(structured_facets)}")
    print("\nSample facets:")
    print(structured_facets.head(10).to_string())
