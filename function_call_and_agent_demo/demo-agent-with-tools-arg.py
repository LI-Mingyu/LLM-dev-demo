import os
import logging
from github import Github
import json
from http import HTTPStatus
from openai import OpenAI
from typing import Tuple, List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('agent_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function definitions remain the same
def get_repo_tree(repo_full_name: str, branch: str | None = None) -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is required")
    
    try:
        g = Github(token)
        repo = g.get_repo(repo_full_name)
    except Exception as e:
        raise RuntimeError(f"Failed to access repository {repo_full_name}: {str(e)}")
    if branch is None:
        branch = "main"
    tree = repo.get_git_tree(sha=branch, recursive=True)
    tree_str = ""
    for item in tree.tree:
        tree_str += f"{item.path}\n"
    return tree_str

def get_repo_file_content(repo_full_name: str, file_path: str, branch: str | None = None) -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is required")
    
    try:
        g = Github(token)
        repo = g.get_repo(repo_full_name)
    except Exception as e:
        raise RuntimeError(f"Failed to access repository {repo_full_name}: {str(e)}")
    if branch is None:
        branch = "main"
    try:
        file_content = repo.get_contents(file_path, ref=branch)
    except Exception as e:
        raise RuntimeError(f"Failed to get file content from {file_path}: {str(e)}")
    return file_content.decoded_content.decode("utf-8")

# Define tools in OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_repo_tree",
            "description": "Get the directory structure of a repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_full_name": {
                        "type": "string",
                        "description": "The full name of the repository, e.g. openai/gpt-3"
                    },
                    "branch": {
                        "type": "string",
                        "description": "The branch name, e.g. master (defaults to 'main' if not provided)"
                    }
                },
                "required": ["repo_full_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_file_content",
            "description": "Get the content of a file in a repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_full_name": {
                        "type": "string",
                        "description": "The full name of the repository, e.g. openai/gpt-3"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file in the repository"
                    },
                    "branch": {
                        "type": "string",
                        "description": "The branch name, e.g. master (defaults to 'main' if not provided)"
                    }
                },
                "required": ["repo_full_name", "file_path"]
            }
        }
    }
]

available_functions = {
    "get_repo_tree": get_repo_tree,
    "get_repo_file_content": get_repo_file_content
}

def call_with_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Make an API call to the LLM with tool support."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is required")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1/"
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    try:
        response = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3",
            # model="qwen-max",
            messages=messages,
            tools=TOOLS,
            temperature=0.5
        )
        return response.choices[0].message
    except Exception as e:
        raise RuntimeError(f"Failed to communicate with LLM: {str(e)}")

def execute_function(tool_call: Dict[str, Any]) -> str:
    """Execute a function based on tool call."""
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    if function_name in available_functions:
        logger.debug(f"Executing function: {function_name} with args: {function_args}")
        return available_functions[function_name](**function_args)
    
    raise ValueError(f"Unknown function: {function_name}")

def run_agent(query: str, max_iterations: int = 10) -> str:
    """Run the agent with the given query."""
    logger.info(f"Starting agent with query: {query}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant for software development tasks."},
        {"role": "user", "content": query}
    ]
    
    iteration = 0
    while iteration < max_iterations:
        try:
            logger.info("Calling LLM with messages:")
            for msg in messages:
                logger.info(f"{msg['role'].capitalize()}: {msg['content']}")
            
            response = call_with_messages(messages)
            logger.info(f"Received LLM response: {response}")
            
            if not response.tool_calls:
                # No tool calls, return the final answer
                logger.info("No tool calls detected, returning final answer")
                return response.content
            
            # Handle tool calls
            if response.tool_calls:
                logger.info(f"Received {len(response.tool_calls)} tool calls")
                for i, tool_call in enumerate(response.tool_calls):
                    logger.info(f"Tool call {i+1}: {tool_call.function.name} with args: {tool_call.function.arguments}")
                
                # First add the assistant message with tool calls
                messages.append({
                    "role": response.role,
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })
                
                # Then add tool responses
                for tool_call in response.tool_calls:
                    function_response = execute_function(tool_call)
                    logger.info(f"Tool execution result for {tool_call.function.name}: {function_response[:200]}...")  # Truncate long responses
                    messages.append({
                        "role": "tool",
                        "content": function_response,
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name
                    })
            
            iteration += 1
            
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {str(e)}")
            return f"Error: {str(e)}"
    
    error_msg = "Agent exceeded maximum iterations without reaching a final answer"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

if __name__ == "__main__":
    try:
        query = "https://github.com/ai-shifu/ChatALL 是如何接入OpenAI的？"
        logger.info(f"Starting agent with query: {query}")
        final_answer = run_agent(query)
        logger.info(f"Agent completed successfully with final answer")
        print(f"\nFinal Answer:\n{final_answer}")
    except Exception as e:
        logger.error(f"Agent failed with error: {str(e)}")
        print(f"Error: {str(e)}")
