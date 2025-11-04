#!/usr/bin/env python3
"""
Demo GRPO Dataset Builder

This demo version shows the complete GRPO dataset generation workflow
without requiring a valid OpenAI API key, using predefined mock responses.
"""

import json
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Mock GRPO responses for demonstration
DEMO_GRPO_RESPONSES = {
    "Чингис хаан": {
        "chosen": "Чингис хаан (1162-1227) бол Монголын агуу хаан, Их Монгол Улсыг байгуулагч юм. Тэрээр 1206 онд Монголын овог аймгуудыг нэгтгэж, дэлхийн түүхэн дэх хамгийн том эзэнт гүрнийг байгуулсан. Чингис хааны удирдлага дор Монголчууд Хятад, Хорезм, Орос зэрэг олон улсыг байлдан дагуулж, Монголын соёл, хууль тогтоомжийг дэлгэрүүлсэн.",
        "rejected": "Чингис хаан бол Монголын хаан байсан. Тэр маш хүчирхэг байсан бөгөөд олон газар байлдсан."
    },
    "1921 оны хувьсгал": {
        "chosen": "1921 оны Монголын ардын хувьсгал нь Монголын түүхэнд чухал үйл явдал болсон. Энэ хувьсгалаар Монгол Улс Манж Чин гүрний засаглалаас ангижирч, социалист замыг сонгосон. Сүхбаатар, Чойбалсан зэрэг удирдагчдын удирдлага дор хувьсгал амжилттай болж, Монголын Ардын Республик байгуулагдсан.",
        "rejected": "1921 онд Монголд хувьсгал болсон. Энэ нь чухал үйл явдал байсан."
    },
    "Богд хааны үе": {
        "chosen": "Богд хааны үе (1911-1924) нь Монголын тусгаар тогтнолын анхны үе байв. VIII Богд Жавзандамба хутагт Монголын хаан болж, теократ засаглал тогтоосон. Энэ үед Монгол Улс Манж Чин гүрнээс тусгаар тогтнож, өөрийн гэсэн засгийн газар, цэрэг байгуулсан боловч олон улсын хүлээн зөвшөөрөл авахад бэрхшээлтэй тулгарсан.",
        "rejected": "Богд хаан Монголын хаан байсан. Тэр 1911-1924 онд засаглаж байсан."
    },
    "Хүннү улс": {
        "chosen": "Хүннү улс (НТӨ 209 - НТ 93) нь Монголын нутагт байгуулагдсан анхны том нүүдэлчдийн улс байв. Модун шаньюйгийн удирдлага дор Хүннү улс хүчирхэгжиж, Хятадын Хан улстай тэнцэхүйц хүчин чадалтай болсон. Хүннүүд нүүдэлчдийн соёл, цэргийн тактик, дипломат харилцааг хөгжүүлж, дараагийн үеийн Монголын улсуудад ихээхэн нөлөө үзүүлсэн.",
        "rejected": "Хүннү улс бол эртний Монголын улс байсан. Тэд нүүдэлчин байсан."
    },
    "Ардчилсан хувьсгал": {
        "chosen": "1990 оны ардчилсан хувьсгал нь Монгол Улсыг нэг намын социалист тогтолцооноос олон намын ардчилсан тогтолцоо руу тайван замаар шилжүүлсэн түүхэн үйл явдал юм. Энэ хувьсгалаар МАХН-ын монополь засаглал дуусч, олон нам үүсэж, 1992 онд шинэ Үндсэн хууль батлагдсан. Монгол Улс зах зээлийн эдийн засагт шилжиж, ардчилсан засаглалыг тогтоосон.",
        "rejected": "1990 онд Монголд ардчилсан хувьсгал болсон. Энэ нь өөрчлөлт авчирсан."
    }
}

class DemoGRPOBuilder:
    """Demo GRPO dataset builder with mock responses."""
    
    def __init__(self):
        """Initialize demo builder."""
        self.question_templates = [
            "{topic} хэзээ болсон бэ?",
            "{topic}-ын үндсэн шалтгаан юу байсан бэ?",
            "{topic}-ын үр дүн нь юу байсан бэ?",
            "{topic} яагаад чухал байсан бэ?",
            "{topic}-д хэн оролцсон бэ?",
            "{topic}-ын ач холбогдол юунд оршдог вэ?",
            "{topic} хэрхэн өрнөсөн бэ?",
            "{topic}-ын талаар дэлгэрэнгүй ярина уу?",
            "{topic} Монголын түүхэнд ямар нөлөө үзүүлсэн бэ?",
            "{topic}-тай холбоотой гол үйл явдлууд юу вэ?"
        ]
    
    def extract_topics_from_content(self, content: str) -> List[str]:
        """Extract topics from content."""
        # Look for key historical terms
        topics = []
        
        # Check for known topics in demo responses
        for topic in DEMO_GRPO_RESPONSES.keys():
            if topic.lower() in content.lower():
                topics.append(topic)
        
        # Extract other potential topics
        topic_patterns = [
            r'(\d{4})\s*оны?\s+([^.!?]{10,50})',
            r'([А-ЯӨҮ][а-яөү\s]{5,30}(?:хувьсгал|дайн|хаан|улс))',
            r'([А-ЯӨҮ][а-яөү\s]{5,30}(?:үе|цаг|зуун))'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    topic = ' '.join(str(m) for m in match if m).strip()
                else:
                    topic = str(match).strip()
                
                topic = re.sub(r'\s+', ' ', topic)
                if 5 < len(topic) < 50:
                    topics.append(topic)
        
        return list(set(topics))[:5]  # Limit to 5 topics per content
    
    def generate_questions_for_topic(self, topic: str, count: int = 2) -> List[str]:
        """Generate questions for a topic."""
        questions = []
        templates = random.sample(self.question_templates, min(count, len(self.question_templates)))
        
        for template in templates:
            question = template.format(topic=topic)
            questions.append(question)
        
        return questions
    
    def generate_grpo_pair(self, question: str, context: str) -> Dict[str, str]:
        """Generate a mock GRPO pair."""
        # Simulate API delay
        time.sleep(0.3)
        
        # Find matching topic in demo responses
        for topic, responses in DEMO_GRPO_RESPONSES.items():
            if topic.lower() in question.lower() or topic.lower() in context.lower():
                return {
                    "prompt": question,
                    "chosen": responses["chosen"],
                    "rejected": responses["rejected"]
                }
        
        # Generate generic response if no match
        return {
            "prompt": question,
            "chosen": f"Энэ асуултын хариулт нь Монголын түүхийн чухал хэсэг юм. Дэлгэрэнгүй мэдээлэл авахын тулд түүхийн эх сурвалжуудыг судлах хэрэгтэй. Энэ үйл явдал Монголын соёл, улс төрийн хөгжилд ихээхэн нөлөө үзүүлсэн байдаг.",
            "rejected": "Энэ талаар тодорхой мэдээлэл байхгүй байна. Магадгүй чухал байсан байх."
        }
    
    def load_source_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load source data from file."""
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError:
                                continue
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        records.append(data)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        return records
    
    def build_demo_grpo_dataset(self, source_file: Path, output_file: Path, pairs_count: int = 10):
        """Build demo GRPO dataset."""
        print("🎯 DEMO: GRPO Dataset Generation")
        print("=" * 50)
        print(f"Source: {source_file}")
        print(f"Output: {output_file}")
        print(f"Target pairs: {pairs_count}")
        print()
        
        # Load source data
        print("📁 Loading source data...")
        records = self.load_source_data(source_file)
        print(f"Loaded {len(records)} records")
        
        # Generate questions
        print("❓ Generating questions from content...")
        question_pairs = []
        
        for record in records[:10]:  # Limit to first 10 records for demo
            content = ""
            for field in ['text', 'content', 'chosen']:
                if field in record and record[field]:
                    content = str(record[field])
                    break
            
            if len(content) < 100:
                continue
            
            topics = self.extract_topics_from_content(content)
            for topic in topics:
                questions = self.generate_questions_for_topic(topic, 1)
                for question in questions:
                    question_pairs.append((question, content))
        
        # Limit to requested count
        question_pairs = question_pairs[:pairs_count]
        print(f"Generated {len(question_pairs)} question-context pairs")
        print()
        
        # Generate GRPO pairs
        print("🔄 Generating GRPO preference pairs...")
        grpo_pairs = []
        
        for question, context in tqdm(question_pairs, desc="Processing"):
            grpo_pair = self.generate_grpo_pair(question, context)
            grpo_pairs.append(grpo_pair)
            
            # Show example
            if len(grpo_pairs) <= 3:
                print(f"\\n✅ Generated pair {len(grpo_pairs)}:")
                print(f"   Prompt: {grpo_pair['prompt']}")
                print(f"   Chosen: {grpo_pair['chosen'][:80]}...")
                print(f"   Rejected: {grpo_pair['rejected'][:60]}...")
        
        # Save results
        print(f"\\n💾 Saving {len(grpo_pairs)} pairs to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in grpo_pairs:
                json.dump(pair, f, ensure_ascii=False)
                f.write('\\n')
        
        # Generate statistics
        chosen_lengths = [len(pair['chosen'].split()) for pair in grpo_pairs]
        rejected_lengths = [len(pair['rejected'].split()) for pair in grpo_pairs]
        prompt_lengths = [len(pair['prompt'].split()) for pair in grpo_pairs]
        
        # Calculate Mongolian purity
        all_text = ' '.join([f"{p['prompt']} {p['chosen']} {p['rejected']}" for p in grpo_pairs])
        mongolian_chars = len(re.findall(r'[А-ЯӨҮа-яөү]', all_text))
        total_chars = len(re.findall(r'[А-ЯӨҮа-яөүA-Za-z]', all_text))
        purity = (mongolian_chars / total_chars * 100) if total_chars > 0 else 0
        
        # Display summary
        print("\\n📊 GRPO DATASET GENERATION REPORT")
        print("=" * 50)
        print(f"Generation Results:")
        print(f"  Total prompts generated: {len(question_pairs)}")
        print(f"  Valid pairs: {len(grpo_pairs)}")
        print(f"  Success rate: 100.0%")
        print()
        print(f"Quality Metrics:")
        print(f"  Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
        print(f"  Average chosen length: {sum(chosen_lengths)/len(chosen_lengths):.1f} words")
        print(f"  Average rejected length: {sum(rejected_lengths)/len(rejected_lengths):.1f} words")
        print(f"  Dataset purity: {purity:.1f}% Mongolian")
        print()
        print(f"Status: ✅ SUCCESS")
        print(f"✅ Ready for GRPO fine-tuning")
        print()
        print("🔧 To use the real GRPO builder:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("2. Run: python scripts/build_grpo_dataset.py")
        
        return output_file

def main():
    """Run demo GRPO dataset generation."""
    builder = DemoGRPOBuilder()
    
    # Use existing dataset
    source_file = Path("data/mgl_history_labeled.jsonl")
    output_file = Path("data/demo_grpo_dataset.jsonl")
    
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        print("Please ensure you have Mongolian historical data available.")
        return 1
    
    try:
        result_file = builder.build_demo_grpo_dataset(source_file, output_file, pairs_count=8)
        print(f"\\n🎉 Demo completed! Check the results in: {result_file}")
        return 0
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())