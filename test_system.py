"""
Test script for the recommendation system
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Recommendation System API")
    print("=" * 50)
    
    # Test 1: Get system stats
    print("\n1. Testing system stats...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ System stats: {stats}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    # Test 2: Get users
    print("\n2. Testing users endpoint...")
    try:
        response = requests.get(f"{base_url}/api/users")
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Found {len(users)} users")
            if users:
                print(f"   Sample user: {users[0]['name']} (ID: {users[0]['user_id']})")
        else:
            print(f"❌ Failed to get users: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting users: {e}")
    
    # Test 3: Get items
    print("\n3. Testing items endpoint...")
    try:
        response = requests.get(f"{base_url}/api/items")
        if response.status_code == 200:
            items = response.json()
            print(f"✅ Found {len(items)} items")
            if items:
                print(f"   Sample item: {items[0]['title']} (ID: {items[0]['item_id']})")
        else:
            print(f"❌ Failed to get items: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting items: {e}")
    
    # Test 4: Get recommendations
    print("\n4. Testing recommendations...")
    try:
        response = requests.get(f"{base_url}/api/recommendations/1?type=hybrid&n=3")
        if response.status_code == 200:
            recs = response.json()
            print(f"✅ Got {len(recs['recommendations'])} recommendations for user 1")
            for rec in recs['recommendations']:
                print(f"   - {rec['item']['title']} (Score: {rec['score']:.3f})")
        else:
            print(f"❌ Failed to get recommendations: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
    
    # Test 5: Add a rating
    print("\n5. Testing add rating...")
    try:
        rating_data = {
            "user_id": 1,
            "item_id": 1,
            "rating": 5
        }
        response = requests.post(f"{base_url}/api/ratings", json=rating_data)
        if response.status_code == 201:
            print("✅ Successfully added rating")
        else:
            print(f"❌ Failed to add rating: {response.status_code}")
    except Exception as e:
        print(f"❌ Error adding rating: {e}")
    
    # Test 6: Get similar items
    print("\n6. Testing similar items...")
    try:
        response = requests.get(f"{base_url}/api/similar/1?n=3")
        if response.status_code == 200:
            similar = response.json()
            print(f"✅ Found {len(similar['similar_items'])} similar items")
            for item in similar['similar_items']:
                print(f"   - {item['item']['title']} (Similarity: {item['similarity']:.3f})")
        else:
            print(f"❌ Failed to get similar items: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting similar items: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("\nTo test the web interface:")
    print("1. Open your browser")
    print("2. Go to http://localhost:5000")
    print("3. Select a user and explore recommendations!")

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    test_api()
