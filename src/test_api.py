"""
Test script for the Clinical Note QA API
Demonstrates how to use the API with sample clinical notes
"""
import requests
import json


API_URL = "http://localhost:8000"


def test_basic_note():
    """Test with a basic clinical note that has several issues"""
    
    note = """
Patient presents with lower back pain. Pain started after lifting heavy box at work yesterday.
Patient says pain is 7/10 and shoots down left leg. Given ibuprofen 600mg and told to rest.
Will follow up in 1 week.
"""
    
    payload = {
        "clinical_note": note,
        "metadata": {
            "note_type": "Initial Evaluation",
            "date_of_service": "2024-01-15",
            "date_of_injury": "2024-01-14"
        }
    }
    
    print("=" * 80)
    print("TEST 1: Basic Note with Multiple Issues")
    print("=" * 80)
    print(f"\nNote:\n{note}\n")
    
    response = requests.post(f"{API_URL}/analyze-note", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Score: {result['score']}/100")
        print(f"Grade: {result['grade']}")
        print(f"\nIssues Found ({len(result['flags'])}):\n")
        
        for i, flag in enumerate(result['flags'], 1):
            print(f"{i}. [{flag['severity'].upper()}]")
            print(f"   Issue: {flag['issue']}")
            print(f"   Suggested Edit: {flag['suggested_edit']}")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_better_note():
    """Test with a more complete clinical note"""
    
    note = """
SUBJECTIVE:
Patient reports acute onset lower back pain beginning 1/14/24 after lifting a 50-pound box at work.
Patient describes pain as sharp, radiating down the left leg to the knee, rated 7/10 in intensity.
Pain is worse with forward bending and prolonged sitting. No bowel or bladder changes reported.

OBJECTIVE:
Physical examination reveals tenderness to palpation over the L4-L5 paraspinal muscles bilaterally.
Positive straight leg raise test on the left at 45 degrees. Range of motion: forward flexion limited 
to 40 degrees (normal 60-90), extension limited to 15 degrees (normal 25). Neurological exam shows 
intact strength in lower extremities, diminished sensation in L5 distribution on left.

ASSESSMENT:
Acute lumbar strain with left-sided radiculopathy, consistent with mechanism of injury sustained 
on 1/14/24. Symptoms and clinical findings support work-relatedness of current condition.

PLAN:
1. Ibuprofen 600mg TID with food for 7 days
2. Physical therapy referral for lumbar stabilization
3. Work restrictions: no lifting >20 lbs, frequent position changes
4. Follow-up in 1 week or sooner if symptoms worsen
"""
    
    payload = {
        "clinical_note": note,
        "metadata": {
            "note_type": "Initial Evaluation",
            "date_of_service": "2024-01-15",
            "date_of_injury": "2024-01-14"
        }
    }
    
    print("\n" + "=" * 80)
    print("TEST 2: Better Structured Note")
    print("=" * 80)
    print(f"\nNote:\n{note}\n")
    
    response = requests.post(f"{API_URL}/analyze-note", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Score: {result['score']}/100")
        print(f"Grade: {result['grade']}")
        print(f"\nIssues Found ({len(result['flags'])}):\n")
        
        for i, flag in enumerate(result['flags'], 1):
            print(f"{i}. [{flag['severity'].upper()}]")
            print(f"   Issue: {flag['issue']}")
            print(f"   Suggested Edit: {flag['suggested_edit']}")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_health_check():
    """Test the health check endpoint"""
    print("=" * 80)
    print("HEALTH CHECK")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("\nClinical Note QA API - Test Script")
    print("Make sure the API is running at http://localhost:8000\n")
    
    try:
        # Test health check first
        test_health_check()
        
        # Test with sample notes
        test_basic_note()
        test_better_note()
        
        print("=" * 80)
        print("Tests completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API at http://localhost:8000")
        print("Please make sure the API is running with: python main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")