"""
Load Testing with Locust

Simulates realistic user traffic for performance testing.

Usage:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10
"""

import random
import string
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner
import logging

logger = logging.getLogger(__name__)


def generate_user_id() -> str:
    """Generate a random user ID."""
    return f"user_{random.randint(1, 10000000)}"


def generate_item_id() -> str:
    """Generate a random item ID."""
    return f"item_{random.randint(1, 1000000)}"


def generate_session_id() -> str:
    """Generate a random session ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))


class RecommendationUser(HttpUser):
    """Simulates a user requesting recommendations."""
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts."""
        self.user_id = generate_user_id()
        self.session_id = generate_session_id()
        self.viewed_items = []
    
    @task(10)
    def get_recommendations(self):
        """Main recommendation request - highest frequency."""
        payload = {
            "user_id": self.user_id,
            "num_recommendations": random.choice([5, 10, 20]),
            "context": {
                "device": random.choice(["mobile", "desktop", "tablet"]),
                "page": random.choice(["home", "category", "search"]),
                "session_id": self.session_id,
            },
            "diversity_factor": random.uniform(0.1, 0.5),
        }
        
        # Add exclude items occasionally
        if self.viewed_items and random.random() < 0.3:
            payload["exclude_items"] = self.viewed_items[-5:]
        
        with self.client.post(
            "/recommend",
            json=payload,
            name="/recommend",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Track viewed items
                if "items" in data:
                    for item in data["items"][:3]:
                        self.viewed_items.append(item.get("item_id"))
                    self.viewed_items = self.viewed_items[-50:]  # Keep last 50
                
                # Validate response
                if "items" not in data:
                    response.failure("Missing 'items' in response")
                elif len(data["items"]) == 0:
                    response.failure("Empty recommendations")
                else:
                    response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(3)
    def get_similar_items(self):
        """Similar items request."""
        item_id = generate_item_id()
        k = random.choice([5, 10, 20])
        
        with self.client.get(
            f"/similar/{item_id}?k={k}",
            name="/similar/{item_id}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "items" in data:
                    response.success()
                else:
                    response.failure("Missing 'items' in response")
            elif response.status_code == 404:
                response.success()  # Item not found is acceptable
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(5)
    def record_click(self):
        """Record click feedback."""
        if not self.viewed_items:
            return
        
        payload = {
            "user_id": self.user_id,
            "item_id": random.choice(self.viewed_items),
            "event_type": "click",
            "context": {
                "position": random.randint(1, 10),
                "page": "recommendations",
            },
        }
        
        with self.client.post(
            "/feedback",
            json=payload,
            name="/feedback",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def record_purchase(self):
        """Record purchase feedback (less frequent)."""
        if not self.viewed_items:
            return
        
        payload = {
            "user_id": self.user_id,
            "item_id": random.choice(self.viewed_items),
            "event_type": "purchase",
            "context": {
                "price": round(random.uniform(10, 500), 2),
                "quantity": random.randint(1, 3),
            },
        }
        
        with self.client.post(
            "/feedback",
            json=payload,
            name="/feedback (purchase)",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Periodic health check."""
        self.client.get("/health", name="/health")


class BatchRecommendationUser(HttpUser):
    """Simulates batch recommendation requests."""
    
    wait_time = between(5, 10)
    weight = 1  # Lower weight than RecommendationUser
    
    @task
    def batch_recommendations(self):
        """Batch recommendation request."""
        num_users = random.choice([5, 10, 20, 50])
        user_ids = [generate_user_id() for _ in range(num_users)]
        
        payload = {
            "user_ids": user_ids,
            "num_recommendations": 10,
            "context": {
                "device": "mobile",
            },
        }
        
        with self.client.post(
            "/recommend/batch",
            json=payload,
            name="/recommend/batch",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure("Missing 'results' in response")
            else:
                response.failure(f"Status {response.status_code}")


class HighVolumeUser(HttpUser):
    """Simulates high-volume API user (e.g., mobile app)."""
    
    wait_time = between(0.1, 0.5)
    weight = 2
    
    def on_start(self):
        self.user_id = generate_user_id()
    
    @task
    def rapid_recommendations(self):
        """Rapid-fire recommendation requests."""
        payload = {
            "user_id": self.user_id,
            "num_recommendations": 10,
        }
        
        self.client.post("/recommend", json=payload, name="/recommend (rapid)")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("Load test starting...")
    
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running as master node")
    elif isinstance(environment.runner, WorkerRunner):
        logger.info("Running as worker node")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    logger.info("Load test completed")
    
    # Print summary statistics
    stats = environment.stats
    
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {stats.total.num_requests:,}")
    print(f"Total Failures: {stats.total.num_failures:,}")
    print(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    print(f"Avg Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"P50 Response Time: {stats.total.get_response_time_percentile(0.5):.2f}ms")
    print(f"P95 Response Time: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"P99 Response Time: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    print("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Called for each request - can be used for custom tracking."""
    if exception:
        logger.warning(f"Request failed: {name} - {exception}")
    elif response_time > 100:  # Log slow requests (>100ms)
        logger.warning(f"Slow request: {name} - {response_time:.0f}ms")
