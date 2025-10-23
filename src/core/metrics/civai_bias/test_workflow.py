#!/usr/bin/env python3
"""
Test script demonstrating the complete civai_bias workflow:
1. Extract factbank from article files
2. Generate brief with LLM
"""

from civai_bias.extraction import create_factbank_from_articles
from civai_bias.brief import generate_brief

def main():
    print("🔄 CivAI Bias Analysis - Complete Workflow Test")
    print("=" * 50)
    
    # Step 1: Extract factbank from articles
    print("\n📰 Step 1: Extracting factbank from article files...")
    print("   - A_left.json (nytimes.com)")
    print("   - B_right.json (foxnews.com)")  
    print("   - C_center.json (reuters.com)")
    
    factbank = create_factbank_from_articles(
        output_path="extracted_factbank.json",
        topic="mail_voting_expansion"
    )
    
    print(f"\n✅ Factbank created with:")
    print(f"   - {len(factbank['core_facts'])} core facts")
    print(f"   - {len(factbank['claims_left'])} left-leaning claims")
    print(f"   - {len(factbank['claims_right'])} right-leaning claims")
    
    # Step 2: Generate brief
    print(f"\n📝 Step 2: Generating brief...")
    
    brief = generate_brief(
        factbank_path="extracted_factbank.json",
        use_all_facts=True,
        use_llm=True
    )
    
    # Step 3: Display results
    print(f"\n📊 Step 3: Results")
    print("=" * 30)
    
    print("\n🔍 FACTS:")
    for fact in brief["facts_bullets"]:
        print(f"  {fact}")
    
    print(f"\n👈 LEFT PERSPECTIVE:")
    for claim in brief["left_bullets"]:
        print(f"  {claim}")
    
    print(f"\n👉 RIGHT PERSPECTIVE:")  
    for claim in brief["right_bullets"]:
        print(f"  {claim}")
    
    if brief["llm_brief"]:
        print(f"\n🤖 AI-GENERATED BRIEF:")
        print("-" * 25)
        print(brief["llm_brief"])
    else:
        print(f"\n❌ LLM brief generation failed")
    
    print(f"\n✅ Workflow completed successfully!")
    print(f"📁 Files created:")
    print(f"   - extracted_factbank.json")
    print(f"   - full_facts.json (detailed extraction data)")

if __name__ == "__main__":
    main()


