#!/usr/bin/env python3
"""
Demonstration script for the Mongolian History Generator.

This script shows how the system would work with a real OpenAI API key.
For demonstration purposes, it creates sample output files.
"""

import os
import json
from datetime import datetime
from pathlib import Path

def create_demo_dataset():
    """Create a demonstration dataset showing the expected output format."""
    
    # Ensure output directory exists
    output_dir = Path("data/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Mongolian History Generator - Demo Mode")
    print("=" * 50)
    print()
    
    # Sample dataset with 5 entries (representing what would be generated)
    sample_dataset = [
        {
            "title": "Establishment of the Bogd Khanate",
            "date": "1911",
            "content": "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\n\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
        },
        {
            "title": "Khiagta Treaty Negotiations",
            "date": "1915",
            "content": "The Treaty of Khiagta established a tripartite agreement between Russia, China, and Mongolia, formally recognizing Mongolian autonomy under Chinese suzerainty. This diplomatic arrangement sought to balance competing imperial interests while providing Mongolia with limited self-governance rights.\n\nThe treaty defined Mongolia's borders and established protocols for Chinese-Mongolian relations, though implementation remained problematic. Russian influence continued to grow during this period, setting the stage for future political developments in the region."
        },
        {
            "title": "Mongolian People's Revolution",
            "date": "1921-03-13",
            "content": "The Mongolian People's Revolution of 1921 brought communist forces to power with Soviet support, establishing the Mongolian People's Party under Damdin Sükhbaatar's leadership. This revolution marked a decisive shift toward socialist governance and closer alignment with the Soviet Union.\n\nThe revolutionary government initially maintained the Bogd Khan as a constitutional monarch while implementing socialist reforms. This period saw the beginning of Mongolia's transformation into a Soviet satellite state, fundamentally altering its political and economic systems."
        },
        {
            "title": "Democratic Revolution of 1990",
            "date": "1990",
            "content": "Mongolia's Democratic Revolution of 1990 marked the peaceful transition from communist rule to a multi-party democratic system. Mass protests and hunger strikes by pro-democracy activists forced the ruling Mongolian People's Revolutionary Party to accept political reforms and competitive elections.\n\nThe revolution led to the adoption of a new constitution in 1992, establishing Mongolia as a parliamentary republic with guaranteed civil liberties and human rights. This transformation made Mongolia one of the first Soviet satellite states to successfully transition to democracy."
        },
        {
            "title": "Mining Boom and Economic Growth",
            "date": "2010",
            "content": "The development of major mining projects, particularly the Oyu Tolgoi copper-gold mine and Tavan Tolgoi coal deposits, transformed Mongolia's economy in the 2010s. These projects attracted billions in foreign investment and positioned Mongolia as a significant player in global commodity markets.\n\nThe mining boom brought rapid economic growth but also created challenges including environmental concerns, income inequality, and economic volatility tied to commodity prices. The government struggled to balance resource extraction with sustainable development and equitable distribution of mining revenues."
        }
    ]
    
    # Write sample dataset
    dataset_file = output_dir / "mongolian_history_dataset_demo.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created demo dataset: {dataset_file}")
    print(f"  - {len(sample_dataset)} historical entries")
    print(f"  - File size: {dataset_file.stat().st_size} bytes")
    
    # Create sample report
    report_data = {
        "generation_summary": {
            "total_topics": 20,
            "successful_generations": 20,
            "failed_generations": 0,
            "total_entries": 65,
            "total_tokens_used": 18450,
            "processing_time_seconds": 342.7,
            "success_rate": 100.0
        },
        "configuration": {
            "model": "gpt-4o-mini",
            "temperature": 0.25,
            "max_tokens": 900,
            "max_retries": 3
        },
        "generation_timestamp": datetime.now().isoformat(),
        "output_files": {
            "dataset_file": "mongolian_history_dataset_demo.json",
            "dataset_size_bytes": dataset_file.stat().st_size,
            "report_file": "generation_report_demo.json"
        },
        "sample_topics_processed": [
            "1911 revolution and the establishment of Bogd Khanate",
            "1915 Khiagta Treaty and Mongolian autonomy", 
            "1921 Mongolian People's Revolution",
            "Democratic Revolution of 1990 in Mongolia",
            "Mining boom: Oyu Tolgoi, Tavan Tolgoi and resource-based growth"
        ]
    }
    
    report_file = output_dir / "generation_report_demo.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created demo report: {report_file}")
    print(f"  - Processing summary and statistics")
    print(f"  - File size: {report_file.stat().st_size} bytes")
    
    print()
    print("Demo Output Summary:")
    print(f"  Total entries: {len(sample_dataset)}")
    print(f"  Date range: 1911-2010")
    print(f"  Average content length: {sum(len(entry['content']) for entry in sample_dataset) // len(sample_dataset)} characters")
    
    print()
    print("To run with real OpenAI API:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Run: python run_generator.py")
    print("  3. Or: python -m mongolian_history_generator")
    
    return dataset_file, report_file

if __name__ == "__main__":
    create_demo_dataset()