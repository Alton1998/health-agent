"""
Example usage of the Health Agent Singleton Pattern

This file demonstrates how to use the singleton pattern to access
the compiled health agent throughout your application.
"""

from agent_singleton import get_health_agent

def main():
    """
    Example of how to use the health agent singleton
    """
    print("=== Health Agent Singleton Example ===\n")
    
    # Get the singleton instance (this will initialize the agent if not already done)
    print("1. Getting health agent instance...")
    agent = get_health_agent()
    
    # Check if agent is initialized
    print(f"2. Agent initialized: {agent.is_initialized()}")
    
    # Get the compiled graph
    print("3. Getting compiled graph...")
    compiled_graph = agent.get_compiled_graph()
    print(f"   Graph type: {type(compiled_graph)}")
    
    # Get model and tokenizer if needed
    print("4. Getting model and tokenizer...")
    model = agent.get_model()
    tokenizer = agent.get_tokenizer()
    print(f"   Model type: {type(model)}")
    print(f"   Tokenizer type: {type(tokenizer)}")
    
    # Demonstrate singleton behavior - getting the same instance
    print("\n5. Demonstrating singleton behavior...")
    agent2 = get_health_agent()
    print(f"   Same instance: {agent is agent2}")
    
    # Example of using the compiled graph
    print("\n6. Example of using the compiled graph...")
    try:
        # Create a sample state for the graph
        sample_state = {
            "messages": [
                {"role": "user", "content": "I have a headache and fever. What could this be? Check my heart rate - it's 90 bpm. Also check my temperature - it's 98.8°F."}
            ],
            "plan": [],
            "task": "symptom_analysis"
        }
        
        print("   Sample state created for graph execution")
        print(f"   State: {sample_state}")
        
        # Note: The actual execution would require proper setup and might need user input
        # This is just to show how you would access the compiled graph
        print("   Compiled graph is ready for execution!")
        
    except Exception as e:
        print(f"   Error during graph setup: {e}")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
