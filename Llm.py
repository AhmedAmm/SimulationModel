import google.generativeai as genai
import os
import time
import random
from dotenv import load_dotenv
from threading import Lock
import threading

load_dotenv()

class RotatingGeminiClient:
    def __init__(self, base_cooldown=60, max_retries=3, exponential_backoff=True):
        self.api_keys = []
        for i in range(1, 7): 
            key = os.getenv(f'api_key{i}')
            if key:
                self.api_keys.append(key)
            else:
                print(f"Warning: api_key{i} not found in .env file")
        
        if not self.api_keys:
            raise ValueError("No API keys found in .env file")
        
        print(f"Loaded {len(self.api_keys)} API keys")
        
        # Rate limiting and retry configuration
        self.base_cooldown = base_cooldown  # Base cooldown in seconds
        self.max_retries = max_retries
        self.exponential_backoff = exponential_backoff
        
        # Track rate limited keys and when they can be used again
        self.rate_limited_keys = {}  # {key_index: timestamp_when_available}
        self.key_lock = Lock()  # Thread safety for key rotation
        
        self.current_key_index = 0
        self._configure_current_key()
    
    def _configure_current_key(self):
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def _is_key_available(self, key_index):
        """Check if a key is available (not rate limited)"""
        if key_index not in self.rate_limited_keys:
            return True
        
        # Check if cooldown period has passed
        current_time = time.time()
        if current_time >= self.rate_limited_keys[key_index]:
            # Remove from rate limited keys
            del self.rate_limited_keys[key_index]
            return True
        
        return False
    
    def _find_available_key(self):
        """Find the next available (non-rate-limited) key"""
        with self.key_lock:
            attempts = 0
            original_index = self.current_key_index
            
            while attempts < len(self.api_keys):
                if self._is_key_available(self.current_key_index):
                    self._configure_current_key()
                    return True
                
                # Move to next key
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                attempts += 1
            
            # If we've tried all keys and none are available
            return False
    
    def _mark_key_rate_limited(self, key_index, retry_attempt=0):
        """Mark a key as rate limited with cooldown period"""
        current_time = time.time()
        
        if self.exponential_backoff:
            # Exponential backoff: base_cooldown * (2 ^ retry_attempt) + random jitter
            cooldown = self.base_cooldown * (2 ** retry_attempt)
            jitter = random.uniform(0, min(cooldown * 0.1, 10))  # Up to 10% jitter, max 10 seconds
            cooldown += jitter
        else:
            # Linear backoff with jitter
            cooldown = self.base_cooldown + (retry_attempt * 30) + random.uniform(0, 10)
        
        available_at = current_time + cooldown
        self.rate_limited_keys[key_index] = available_at
        
        print(f"API key {key_index + 1} rate limit exceeded. Available again in {cooldown:.1f} seconds")
    
    def _is_rate_limit_error(self, error):
        """Check if the error is related to rate limiting"""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit',
            'quota exceeded',
            'too many requests',
            'resource_exhausted',
            '429',
            'rate_limit_exceeded'
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _get_clean_error_message(self, error):
        """Return a clean, simplified error message"""
        error_str = str(error)
        
        # Check for rate limit errors
        if self._is_rate_limit_error(error):
            return "Rate limit exceeded"
        
        # Check for other common errors
        if "400" in error_str or "invalid" in error_str.lower():
            return "Invalid request"
        elif "401" in error_str or "unauthorized" in error_str.lower():
            return "Authentication failed"
        elif "403" in error_str or "forbidden" in error_str.lower():
            return "Access forbidden"
        elif "500" in error_str or "internal" in error_str.lower():
            return "Server error"
        elif "timeout" in error_str.lower():
            return "Request timeout"
        elif "network" in error_str.lower() or "connection" in error_str.lower():
            return "Network error"
        else:
            # For any other error, return a generic message
            return "API error occurred"
    
    def _wait_for_available_key(self):
        """Wait until at least one key becomes available"""
        if not self.rate_limited_keys:
            return
        
        # Find the key that will be available soonest
        current_time = time.time()
        min_wait_time = min(self.rate_limited_keys.values()) - current_time
        
        if min_wait_time > 0:
            print(f"All keys rate limited. Waiting {min_wait_time:.1f} seconds for next available key...")
            time.sleep(min_wait_time + 1)  # Add 1 second buffer
    
    def ask_gemini(self, prompt, thread_id=None):
        """Send a prompt to Gemini with infinite retry until successful response"""
        thread_prefix = f"[Thread {thread_id}] " if thread_id else ""
        
        while True:  # Infinite loop until successful response
            try:
                # Find an available key
                if not self._find_available_key():
                    print(f"{thread_prefix}All API keys are rate limited. Waiting...")
                    self._wait_for_available_key()
                    time.sleep(30)
                    continue  # Go back to beginning of while loop
                
                print(f"{thread_prefix}Attempting request using API key {self.current_key_index + 1}")
                
                # Make the request
                response = self.model.generate_content(prompt)
                
                # Check if response has valid content
                if response and hasattr(response, 'text') and response.text:
                    result = response.text.strip()
                    if result:  # Only return if we have actual content
                        print(f"{thread_prefix}Success with API key {self.current_key_index + 1}")
                        return result
                    else:
                        print(f"{thread_prefix}Empty response received, retrying...")
                        time.sleep(random.uniform(1, 3))
                        continue
                else:
                    print(f"{thread_prefix}Invalid response received, retrying...")
                    time.sleep(random.uniform(1, 3))
                    continue
                
            except Exception as e:
                # Get clean error message
                clean_error = self._get_clean_error_message(e)
                print(f"{thread_prefix}Error with API key {self.current_key_index + 1}: {clean_error} - Retrying...")
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    self._mark_key_rate_limited(self.current_key_index)
                
                # Add a small delay before retry to avoid hammering the API
                time.sleep(random.uniform(1, 3))
                continue  # Go back to beginning of while loop
    
    def get_status(self):
        """Get current status of all API keys"""
        current_time = time.time()
        status = {
            'total_keys': len(self.api_keys),
            'current_key': self.current_key_index + 1,
            'rate_limited_keys': len(self.rate_limited_keys),
            'available_keys': len(self.api_keys) - len(self.rate_limited_keys)
        }
        
        if self.rate_limited_keys:
            status['rate_limited_details'] = {}
            for key_index, available_at in self.rate_limited_keys.items():
                time_remaining = max(0, available_at - current_time)
                status['rate_limited_details'][f'key_{key_index + 1}'] = f"{time_remaining:.1f}s remaining"
        
        return status


# Example usage with threading
def worker_thread(client, prompts, thread_id):
    """Worker function for threading example"""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Thread {thread_id} processing prompt {i+1}/{len(prompts)}")
        result = client.ask_gemini(prompt, thread_id=thread_id)
        results.append(result)
        
        # Small delay between requests in the same thread
        time.sleep(random.uniform(0.5, 1.5))
    
    return results

# Initialize the client with custom settings
llm = RotatingGeminiClient(
    base_cooldown=60,      # Base cooldown of 60 seconds for rate limited keys
    max_retries=3,         # This parameter is no longer used since we retry infinitely
    exponential_backoff=True  # Use exponential backoff for cooldowns
)

# Example of using with multiple threads:
# import threading
# 
# prompts_batch_1 = ["Tell me about AI", "Explain quantum computing", "What is machine learning?"]
# prompts_batch_2 = ["Describe blockchain", "Explain neural networks", "What is deep learning?"]
# 
# thread1 = threading.Thread(target=worker_thread, args=(llm, prompts_batch_1, 1))
# thread2 = threading.Thread(target=worker_thread, args=(llm, prompts_batch_2, 2))
# 
# thread1.start()
# thread2.start()
# 
# thread1.join()
# thread2.join()
# 
# print("Status:", llm.get_status())