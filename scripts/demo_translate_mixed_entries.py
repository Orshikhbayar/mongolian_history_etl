#!/usr/bin/env python3
"""
Demo version of Mongolian Dataset Translation Script

This demo version simulates successful translations to show the complete workflow
without requiring a valid OpenAI API key.
"""

import json
import re
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

# Import the main classes from the original script
import sys
sys.path.append('scripts')

# Mock translations for demo purposes
DEMO_TRANSLATIONS = {
    "Mining industry became crucial for Mongolia's economy in the 2000s. The Oyu Tolgoi copper mine project attracted billions in foreign investment.": 
    "Уул уурхайн салбар 2000-аад оноос эхлэн Монголын эдийн засгийн гол тулгуур болсон. Оюу толгой зэс уурхайн төсөл олон тэрбум долларын гадаадын хөрөнгө оруулалт татсан.",
    
    "COVID-19 pandemic significantly impacted Mongolia in 2020-2022. The government implemented strict border controls and lockdown measures.":
    "КОВИД-19 цар тахал 2020-2022 онд Монгол Улсад ихээхэн нөлөө үзүүлсэн. Засгийн газар хатуу хилийн хяналт болон хөл хорио арга хэмжээ авч хэрэгжүүлсэн.",
    
    "The Democratic Revolution of 1990 peacefully transformed Mongolia from a one-party socialist state to a multi-party democracy.":
    "1990 оны ардчилсан хувьсгал нь Монгол Улсыг нэг намын социалист улсаас олон намын ардчилсан улс болгон тайван замаар өөрчилсөн.",
    
    "Genghis Khan was a Mongolian leader who founded the Mongol Empire.":
    "Чингис хаан бол Монголын удирдагч байсан бөгөөд Их Монгол Улсыг байгуулсан."
}

class MockMongolianTranslator:
    """Mock translator that uses predefined translations."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def translate_text(self, text: str) -> Tuple[str, bool, Dict[str, Any]]:
        """Mock translation using predefined translations."""
        # Simulate API delay
        time.sleep(0.5)
        
        # Check if we have a demo translation
        if text in DEMO_TRANSLATIONS:
            translated = DEMO_TRANSLATIONS[text]
            metadata = {
                'tokens_used': len(text.split()) * 2,  # Simulate token usage
                'api_calls': 1
            }
            return translated, True, metadata
        else:
            # For unknown text, create a simple mock translation
            # Replace English words with Mongolian equivalents
            mock_translation = text.replace("Mongolia", "Монгол Улс")
            mock_translation = mock_translation.replace("Mongolian", "Монголын")
            mock_translation = mock_translation.replace("the", "")
            mock_translation = mock_translation.replace("and", "ба")
            mock_translation = mock_translation.replace("in", "дотор")
            mock_translation = mock_translation.replace("of", "")
            
            metadata = {
                'tokens_used': len(text.split()) * 2,
                'api_calls': 1
            }
            return mock_translation, True, metadata

def demo_translation():
    """Run a demo translation."""
    print("🎯 DEMO: Mongolian Dataset Translation")
    print("=" * 50)
    print("This demo shows how the translation script works with successful API calls.")
    print()
    
    # Create demo input file
    demo_data = [
        {
            "text": "1990 оны ардчилсан хувьсгал нь Монгол Улсын түүхэнд чухал үйл явдал болсон.",
            "period": "XX зуун",
            "source": "Demo Dataset"
        },
        {
            "text": "Mining industry became crucial for Mongolia's economy in the 2000s. The Oyu Tolgoi copper mine project attracted billions in foreign investment.",
            "period": "XXI зуун", 
            "source": "Demo Dataset"
        },
        {
            "text": "COVID-19 pandemic significantly impacted Mongolia in 2020-2022. The government implemented strict border controls and lockdown measures.",
            "period": "XXI зуун",
            "source": "Demo Dataset"
        },
        {
            "text": "Монголын гадаад бодлого нь 'гуравдахь хөрш' бодлогод суурилдаг.",
            "period": "XXI зуун",
            "source": "Demo Dataset"
        }
    ]
    
    demo_input_path = Path("data/demo_mixed_dataset.json")
    demo_output_path = Path("data/demo_translated.jsonl")
    
    # Save demo input
    with open(demo_input_path, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)
    
    print(f"📁 Created demo input: {demo_input_path}")
    print(f"📄 Input records: {len(demo_data)}")
    print()
    
    # Simulate the translation process
    print("🔄 Processing records...")
    
    # Language detection
    english_pattern = re.compile(r'[A-Za-z]')
    mongolian_pattern = re.compile(r'[А-ЯӨҮа-яөү]')
    
    translated_records = []
    stats = {
        'total_records': len(demo_data),
        'mixed_detected': 0,
        'translated': 0,
        'skipped': 0,
        'tokens_used': 0
    }
    
    translator = MockMongolianTranslator()
    
    for i, record in enumerate(tqdm(demo_data, desc="Translating")):
        text = record.get('text', '')
        
        # Analyze language composition
        english_chars = len(english_pattern.findall(text))
        mongolian_chars = len(mongolian_pattern.findall(text))
        total_alpha = english_chars + mongolian_chars
        
        if total_alpha > 0:
            english_ratio = english_chars / total_alpha
        else:
            english_ratio = 0
        
        # Check if translation needed (>20% English)
        if english_ratio >= 0.2:
            stats['mixed_detected'] += 1
            
            # Translate
            translated_text, success, metadata = translator.translate_text(text)
            
            if success:
                record_copy = record.copy()
                record_copy['text'] = translated_text
                translated_records.append(record_copy)
                stats['translated'] += 1
                stats['tokens_used'] += metadata['tokens_used']
                
                print(f"✅ Translated record {i+1}")
                print(f"   Original: {text[:60]}...")
                print(f"   Translated: {translated_text[:60]}...")
                print()
        else:
            # Already Mongolian, keep as-is
            translated_records.append(record)
            stats['skipped'] += 1
            print(f"⏭️ Skipped record {i+1} (already Mongolian)")
    
    # Save results
    with open(demo_output_path, 'w', encoding='utf-8') as f:
        for record in translated_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\\n')
    
    print(f"💾 Saved results to: {demo_output_path}")
    print()
    
    # Calculate final purity
    final_purity = 100.0  # All translations successful in demo
    
    # Generate summary report
    print("📊 TRANSLATION SUMMARY")
    print("=" * 30)
    print(f"Total records: {stats['total_records']}")
    print(f"Mixed-language detected: {stats['mixed_detected']}")
    print(f"Successfully translated: {stats['translated']}")
    print(f"Skipped (already Mongolian): {stats['skipped']}")
    print(f"Final Mongolian purity: {final_purity:.1f}%")
    print(f"Total tokens used: {stats['tokens_used']:,}")
    print()
    print("✅ Demo completed successfully!")
    print()
    print("🔧 To use the real translation script:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Run: python scripts/translate_mixed_entries.py")
    
    return demo_output_path

if __name__ == "__main__":
    demo_translation()