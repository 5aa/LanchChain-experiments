from typing import Any, Dict

from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
)

# TODO: modify to use ConversationSummaryBufferMemory, override load_memory_variables method
class ChatMLFormatMemory(ConversationBufferMemory):
    """Modified buffer for storing conversation memory with ChatML formatting."""

    human_id: str = "user"
    ai_id: str = "assistant"
    memory_key: str = "chat_history"

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a ChatML formatted string"""
        string_messages = []
        for m in self.chat_memory.messages:
            if isinstance(m, HumanMessage):
                role = self.human_id
            elif isinstance(m, AIMessage):
                role = self.ai_id
            elif isinstance(m, SystemMessage):
                role = "System"
            elif isinstance(m, FunctionMessage):
                role = "Function"
            elif isinstance(m, ChatMessage):
                role = m.role
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            message = f"<|im_start|>{role}\n{m.content}<|im_end|>"
            if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
                message += f"{m.additional_kwargs['function_call']}"
            string_messages.append(message)

        return ("\n".join(string_messages) + "\n") if self.chat_memory.messages else ""
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Overriding this method to handle the ": " at the end of OpenOrca outputs."""
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str.lstrip(": "))