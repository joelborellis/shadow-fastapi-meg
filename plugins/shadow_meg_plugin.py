from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from openai import AsyncOpenAI


class ShadowMegPlugin:
    """Plugin class that handles MEG actions and tasks."""

    def __init__(
        self, openai_client: AsyncOpenAI
    ):
        """
        :param openai_client: openai client
        """
        
        self.openai_client = openai_client
        
    @kernel_function(
        name="summarize_conversation",
        description="When the user asks for a summary of the conversation, retrieve the current conversation with the threadId and summarize it.",
    )
    def summarize_conversation(
        self, query: Annotated[str, "The query from the user."], currentThreadId: Annotated[str, f"The currentThreadId."]
    ) -> Annotated[str, "Returns a summary of the current conversation."]:
        try:
            print(f"query:  {query} thread_id:  {currentThreadId}")
            response = self.openai_client.beta.threads.messages.list(currentThreadId)
            
            if response:
                for msg in response:
                    print(f"{msg['role']}: {msg['content']}")
            
            #messages = response.data
        
            #sorted_messages = sorted(messages, key=lambda x: x.get('created_at', 0))


        except Exception as e:
            print(f"Error retrieving thread history: {e}")
            return None