import asyncio
from dotenv import load_dotenv
from loguru import logger

# Initialize environment variables
load_dotenv()

from src.agent.graph import build_agent_graph

async def main():
    logger.info("Initializing Agentic Graph Core Execution System...")
    
    # Compile graph infrastructure
    app = build_agent_graph()
    
    # Define a test question
    # Pro-tip: Ask something related to the test document you uploaded in Step 5!
    user_query = "What is the content discussed in the documentation?"
    
    initial_state = {
        "query": user_query,
        "retrieved_docs": [],
        "generation": "",
        "steps": []
    }
    
    logger.info(f"Invoking graph loop with target query: '{user_query}'")
    
    # Run the compiled graph asynchronously
    final_output = await app.ainvoke(initial_state)
    
    print("\n" + "="*50)
    print("🤖 AGENT FINAL SYSTEM RESPONSE:")
    print("="*50)
    print(final_output.get("generation", "No generation produced."))
    print("="*50)
    print(f"Graph Path Steps Traced: {final_output.get('steps')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())