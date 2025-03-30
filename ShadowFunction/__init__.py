import fastapi
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import AsyncOpenAI

from pydantic import BaseModel
import os
import logging

from semantic_kernel.kernel import Kernel
from semantic_kernel.agents.open_ai import OpenAIAssistantAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

# Import the modified plugin class
from plugins.shadow_meg_plugin import ShadowMegPlugin

from typing import Optional

app = fastapi.FastAPI()

# Allow requests from all domains (not always recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure module-level logger
logger = logging.getLogger("__init__.py")
logger.setLevel(logging.INFO)


# Define request body model
class ShadowRequest(BaseModel):
    query: str
    threadId: str

# Instantiate search clients as singletons (if they are thread-safe or handle concurrency internally)
openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
ASSISTANT_ID = os.environ.get("ASSISTANT_ID")
currentthreadId = ""

async def get_agent() -> Optional[OpenAIAssistantAgent]:
    """
    Setup the Assistant with error handling.
    """
    try:
        # (1) Create the instance of the Kernel
        kernel = Kernel()
    except Exception as e:
        logger.error("Failed to initialize the kernel: %s", e)
        return None
    
    try:
        # (2) Add plugin
        # Instantiate ShadowMegPlugin and pass the openai client
        shadow_plugin = ShadowMegPlugin(openai_client)
    except Exception as e:
        logger.error("Failed to instantiate ShadowMegPlugin: %s", e)
        return None

    try:
        # (3) Register plugin with the Kernel
        kernel.add_plugin(shadow_plugin, plugin_name="shadowMegPlugin")
    except Exception as e:
        logger.error("Failed to register plugin with the kernel: %s", e)
        return None

    try:
        # (4) Retrieve the agent
        agent = await OpenAIAssistantAgent.retrieve(
            id=ASSISTANT_ID, kernel=kernel, ai_model_id="gpt-4.5-preview"
        )
        if agent is None:
            logger.error("Failed to retrieve the assistant agent. Please check the assistant ID.")
            return None
    except Exception as e:
        logger.error("An error occurred while retrieving the assistant agent: %s", e)
        return None

    return agent

@app.post("/meg_chat")
async def meg_chat(request: ShadowRequest):
    """
    Endpoint that receives a query, passes it to the agent, and returns a single JSON response.
    """
    agent = await get_agent()
    if agent is None:
        return {"error": "Failed to retrieve the assistant agent."}
    
    # Retrieve or create a thread ID
    if request.threadId:
        currentthreadId = request.threadId
    else:
        currentthreadId = await agent.create_thread()

    # Create the user message content with the request.query
    message_user = ChatMessageContent(role=AuthorRole.USER, content=request.query)
    await agent.add_chat_message(thread_id=currentthreadId, message=message_user)

    try:
        # Collect all messages from the async iterable
        full_response = []
        async for message in agent.invoke(thread_id=currentthreadId):
            if message.content.strip():  # Skip empty content
                full_response.append(message.content)

        if not full_response:
            return {"error": "Empty response from the agent.", "thread_id": currentthreadId}

        # Combine the collected messages into a single string
        combined_response = " ".join(full_response)

        json_response = {
            "data": combined_response,
            "thread_id": currentthreadId
        }
        # Return json response
        return JSONResponse(json_response, status_code=200)

    except HTTPException as exc:
        return {"error": exc.detail}
    except Exception as e:
        logging.exception("Unexpected error during response generation.")
        return {"error": str(e)}