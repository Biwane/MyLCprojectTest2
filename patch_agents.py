from langchain.tools.base import Tool

# Create a simple dummy tool
dummy_tool = Tool(
    name="dummy_tool",
    description="A dummy tool that does nothing",
    func=lambda x: "This tool does nothing"
)

# Create a list of tools
tools = [dummy_tool]

# Import the agent factory
from core.agent_factory import AgentFactory

# Save the original method
original_create_agent = AgentFactory.create_agent

# Define a patched version
def patched_create_agent(self, *args, **kwargs):
    # Add tools to kwargs
    if 'tools' not in kwargs:
        kwargs['tools'] = tools
    return original_create_agent(self, *args, **kwargs)

# Apply the patch
AgentFactory.create_agent = patched_create_agent

print("Agent factory patched successfully!")
