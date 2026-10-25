import os
import sys
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def initialize_model(
    model_name: str = "qwen3-coder:latest",
    temperature: float = 0.8,
    num_predict: int = 1024,
    base_url: Optional[str] = None
) -> Optional[ChatOllama]:
    """Initialize Qwen model locally through Ollama with error handling."""
    try:
        # Use environment variable for base URL if not provided
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Try with request_timeout first (newer versions)
        try:
            model = ChatOllama(
                model=model_name,
                temperature=temperature,
                num_predict=num_predict,
                base_url=base_url,
                request_timeout=300  # 5 minutes timeout
            )
        except TypeError:
            # Fallback for older versions that use 'timeout'
            model = ChatOllama(
                model=model_name,
                temperature=temperature,
                num_predict=num_predict,
                base_url=base_url,
                timeout=300
            )
        
        # Test the connection with a simple prompt
        test_response = model.invoke([HumanMessage(content="Hello")])
        if test_response:
            print(f"✓ Successfully connected to model: {model_name}")
            return model
        else:
            print(f"✗ Model {model_name} returned empty response")
            return None
            
    except Exception as e:
        print(f"✗ Error initializing model {model_name}: {e}")
        print("Please ensure:")
        print("1. Ollama is running locally")
        print(f"2. Model '{model_name}' is available (run: ollama pull {model_name})")
        print(f"3. Ollama is accessible at {base_url}")
        return None


def generate_response(model: ChatOllama, prompt: str) -> Optional[str]:
    """Generate response with proper error handling."""
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"✗ Error generating response: {e}")
        return None


def main():
    """Main execution function."""
    # Initialize the model
    model = initialize_model()
    
    if model is None:
        print("Failed to initialize model. Exiting.")
        sys.exit(1)
    
    # Define your prompt
    prompt = "Write a short story about a robot learning to paint."
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    print("-" * 50)
    
    # Generate the response
    response = generate_response(model, prompt)
    
    if response:
        print(response)
    else:
        print("Failed to generate response.")
        sys.exit(1)


if __name__ == "__main__":
    main()