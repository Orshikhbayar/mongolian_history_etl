#!/usr/bin/env python3
"""
Setup OpenAI API Key and Create Embeddings
This script helps you set a valid API key and create embeddings for the RAG system.
"""

import os
import sys
from pathlib import Path
from getpass import getpass


def test_api_key(api_key: str) -> bool:
    """Test if the API key is valid."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a simple embedding request
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        return True
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False


def save_api_key_to_env(api_key: str):
    """Save API key to .env file."""
    env_file = Path('.env')
    
    # Read existing .env or create new
    env_lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_lines = [line for line in f.readlines() if not line.startswith('OPENAI_API_KEY=')]
    
    # Add new API key
    env_lines.append(f'OPENAI_API_KEY={api_key}\n')
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(env_lines)
    
    print(f"✅ API key saved to {env_file}")


def create_embeddings_with_filtered_dataset():
    """Create embeddings using the filtered dataset."""
    print("\n🔧 Creating embeddings with filtered dataset...")
    
    # Update the embedding pipeline to use filtered dataset
    from mongolian_rag.embedding_pipeline import EmbeddingPipeline
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found in environment")
        return False
    
    try:
        # Initialize pipeline
        embedder = EmbeddingPipeline(
            openai_api_key=api_key,
            model='text-embedding-3-small',
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Create embeddings from filtered dataset
        print("📊 Processing filtered dataset...")
        index, metadata = embedder.create_embeddings(
            'data/mongolian_history_unified_filtered.jsonl'
        )
        
        # Save to disk
        output_dir = Path('data/embeddings_filtered')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        embedder.save_pipeline(str(output_dir))
        
        print(f"✅ Embeddings created and saved to {output_dir}")
        print(f"📈 Total chunks: {len(metadata)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        return False


def main():
    """Main setup flow."""
    print("🔑 OpenAI API Key Setup & Embedding Creation")
    print("=" * 60)
    
    # Check if API key is already set and valid
    current_key = os.getenv('OPENAI_API_KEY')
    
    if current_key:
        print(f"\n📋 Current API key found: {current_key[:20]}...")
        print("🧪 Testing current API key...")
        
        if test_api_key(current_key):
            print("✅ Current API key is valid!")
            use_current = input("\nUse current key? (y/n): ").strip().lower()
            
            if use_current == 'y':
                # Proceed with current key
                return create_embeddings_with_filtered_dataset()
    
    # Get new API key
    print("\n🔑 Please enter your OpenAI API key")
    print("Get your key from: https://platform.openai.com/api-keys")
    print("-" * 60)
    
    while True:
        api_key = getpass("OpenAI API Key: ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty")
            continue
        
        if not api_key.startswith('sk-'):
            print("❌ Invalid API key format (should start with 'sk-')")
            continue
        
        # Test the key
        print("\n🧪 Testing API key...")
        if test_api_key(api_key):
            print("✅ API key is valid!")
            
            # Set in environment
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Ask to save
            save = input("\nSave API key to .env file? (y/n): ").strip().lower()
            if save == 'y':
                save_api_key_to_env(api_key)
            
            # Create embeddings
            create = input("\nCreate embeddings now? (y/n): ").strip().lower()
            if create == 'y':
                return create_embeddings_with_filtered_dataset()
            else:
                print("\n✅ API key set successfully!")
                print("Run this script again or use: python scripts/setup_rag_pipeline.py")
                return True
        else:
            retry = input("\nTry another key? (y/n): ").strip().lower()
            if retry != 'y':
                return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
