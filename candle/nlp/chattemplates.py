from typing import List, Tuple
from abc import ABC, abstractmethod


class ChatTemplate(ABC):
    """Chat templates format chat messages into a format conducive to chat modelling."""
    
    def apply_chat_template(self,
                            messages: List[Tuple[str, str]],
                            add_generation_prompt: bool = False,
                            n_recent_messages: int = None):
        """Applies chat template.
        
        Parameters
        ----------
        messages
            List of (username, message).
        add_generation_prompt
            If True, then adds a prompt at the end indicating the model should begin generation.
        n_recent_messages
            If True, then filters to the `n_recent_messages` most recent messages. This is useful for
            cumulative KV-cached decoding.
            
            Concretely, returns `s` such that:
                apply_chat_template(messages[:-n_recent_messages]) + `s` = apply_chat_template(messages)
            
            if n_recent_messages >= len(messages), simply returns apply_chat_template(messages).

        """
        full_chat = self._apply_chat_template(messages, add_generation_prompt)

        if n_recent_messages is not None and n_recent_messages < len(messages):
            if n_recent_messages <= 0:
                raise ValueError(f'n_recent_messages = {n_recent_messages} must be > 0.')

            partial_chat = self._apply_chat_template(messages[:-n_recent_messages], add_generation_prompt)

            if not full_chat.startswith(partial_chat):
                raise ValueError(f'full chat = \n"""{full_chat}""" \n\nmust start with '
                                 f'partial_chat = \n"""{partial_chat}""".')

            # `diff` satisfies `partial_chat + diff == full_chat`
            diff = full_chat[len(partial_chat):]
            assert partial_chat + diff == full_chat
            
            return diff
        
        else:
            return full_chat

    
    @abstractmethod
    def _apply_chat_template(self,
                             messages: List[Tuple[str, str]],
                             add_generation_prompt: bool = False):
        pass
    
    
class SimpleConversationTemplate(ChatTemplate):
    """Simple conversation chat template.

    Examples
    --------
    >>> messages = [
    ...     {'role': 'user', 'content': 'Hi, how are you?'},
    ...     {'role': 'assistant', 'content': 'I am doing well. How about you?'},
    ...     {'role': 'user', 'content': 'Great thanks!'},
    ...     {'role': 'assistant', 'content': 'Nice!'},
    ...     {'role': 'user', 'content': 'What is your favourite baseball team?'},
    ... ]

    >>> chat_template = SimpleConversationTemplate(user_name='John', asst_name='Taylor')
    >>> print(chat_template.apply_chat_template(messages, add_generation_prompt=True))

    Two friends, John and Taylor, are having an online conversation. Taylor is friendly,
    talkative, and loves to ask John questions.

    John: Hi, how are you?
    Taylor: I am doing well. How about you?
    John: Great thanks!
    Taylor: Nice!
    John: What is your favourite baseball team?
    Taylor:

    >>> print(chat_template.apply_chat_template(messages, add_generation_prompt=True,
                                                n_recent_messages=2))

     Nice!
    John: What is your favourite baseball team?
    Taylor:
    
    """
    
    def __init__(self,
                 user_name: str,
                 asst_name: str,
                 system_message = ('Two friends, {user_name} and {asst_name}, are having '
                                   'an online conversation. {asst_name} is friendly, '
                                   'talkative, and loves to ask {user_name} questions.')):
        self.name_by_role = {
            'user': user_name,
            'assistant': asst_name
        }
        
        self.system_message = system_message.format(user_name=user_name, asst_name=asst_name)
        
        
    def _apply_chat_template(self,
                             messages: List[Tuple[str, str]],
                             add_generation_prompt: bool):
        chat = self.system_message + '\n\n'

        for message in messages:
            chat += f'{self.name_by_role[message["role"]]}: {message["content"]}\n'
            
        if add_generation_prompt:
            chat += f'{self.name_by_role["assistant"]}:'

        return chat
    

class ChatML(ChatTemplate):
    """OpenAI's Chat Markup Language (*I think... I haven't found that much documentation online)

    Examples
    --------
    >>> messages = [
    ...     {'role': 'user', 'content': 'Hi, how are you?'},
    ...     {'role': 'assistant', 'content': 'I am doing well. How about you?'},
    ...     {'role': 'user', 'content': 'Great thanks!'},
    ...     {'role': 'assistant', 'content': 'Nice!'},
    ...     {'role': 'user', 'content': 'What is your favourite baseball team?'},
    ... ]

    >>> chat_template = chat_template = ChatML(system_message='You are a helpful assistant.')
    >>> print(chat_template.apply_chat_template(messages, add_generation_prompt=True))

    <|im_start|>system
    You are a helpful assistant.
    <|im_start|>user
    Hi, how are you?
    <|im_start|>assistant
    I am doing well. How about you?
    <|im_start|>user
    Great thanks!
    <|im_start|>assistant
    Nice!
    <|im_start|>user
    What is your favourite baseball team?
    <|im_start|>assistant

    >>> print(chat_template.apply_chat_template(messages, add_generation_prompt=True,
                                                n_recent_messages=2))

    Nice!
    <|im_start|>user
    What is your favourite baseball team?
    <|im_start|>assistant
    
    """
    
    def __init__(self,
                 system_message: str):
        
        self.system_message = system_message
        
        
    def _apply_chat_template(self,
                             messages: List[Tuple[str, str]],
                             add_generation_prompt: bool):
        chat = '<|im_start|>system'
        chat += f'\n{self.system_message}'

        for message in messages:
            chat += f'\n<|im_start|>{message["role"]}'
            chat += f'\n{message["content"]}'

        if add_generation_prompt:
            chat += f'\n<|im_start|>assistant\n'

        return chat
