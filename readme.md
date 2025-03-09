# Team Agents System

A flexible framework for creating and managing dynamic teams of AI agents that collaborate to solve complex tasks.

## Overview

This system enables the creation of specialized AI agent teams that work together to accomplish tasks. The framework:

1. Analyzes a task to determine the required team composition
2. Assembles a team of specialized agents (researchers, planners, specialists, executors, reviewers)
3. Coordinates the agents' work to complete the task
4. Produces consolidated results and artifacts

## Directory Structure

```
team_agents/
├── agents/           # Agent implementations for different roles
├── core/             # Core system components
├── memory/           # Memory and persistence components
├── output/           # Generated outputs and artifacts
├── tools/            # Tools used by agents
├── utils/            # Utility functions and helpers
├── data/             # Persistent data storage
├── config.yaml       # System configuration
├── main.py           # Main entry point
└── README.md         # This file
```

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
4. Run the setup script to create necessary directories:
   ```
   python create_directories.py
   ```

## Usage

You can use the system in two ways:

### 1. Command Line Interface

```
python main.py "Your task description here"
```

### 2. Interactive Mode

```
python main.py --interactive
```

### Example

Try running the example script:

```
python example_usage.py
```

## Configuration

The system is configured through `config.yaml`. You can modify this file to customize:

- Models used for different agent roles
- Task scheduling parameters
- Knowledge repository settings
- Tool configurations

## Extending the System

### Adding New Agent Types

Create a new agent class in the `agents/` directory that inherits from `BaseAgent`.

### Adding New Tools

Implement new tools in the `tools/` directory following the existing pattern.

## License

[Specify license information here]
