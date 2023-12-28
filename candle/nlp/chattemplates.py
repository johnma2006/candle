from typing import List, Tuple
from abc import ABC, abstractmethod


class ChatTemplate(ABC):
    """Chat templates format chat messages into a format conducive to chat modelling."""
    
    @abstractmethod
    def apply_chat_template(self,
                            messages: List[Tuple[str, str]],
                            add_generation_prompt: bool = False,
                            n_recent_messages: int = None):
        """Applies chat template.
        
        Args:
            messages: List of (username, message).
            add_generation_prompt (bool): If True, then adds a prompt at the end indicating
                the model should begin generation.
            n_recent_messages (bool): If True, then filters to the `n_recent_messages` most
                recent messages. This is useful for cumulative KV-cached decoding.
                
                Concretely, returns `s` such that:
                    apply_chat_template(messages[:-n_recent_messages]) + `s` = apply_chat_template(messages)
                
                if n_recent_messages >= len(messages), simply returns apply_chat_template(messages).

        """
        pass
        
    
class SimpleConversationTemplate(ChatTemplate):
    """Simple conversation chat template.

    Examples:
        >>> messages = [
        ...     {'role': 'system', 'content': self.system_message},
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
        
        
    def apply_chat_template(self,
                            messages: List[Tuple[str, str]],
                            add_generation_prompt: bool):
        chat = ''
        for message in messages:
            if message['role'] == 'system':
                chat += message['content'] + '\n'
            else:
                chat += f'\n{self.name_by_role[message["role"]]}: {message["content"]}'

        if add_generation_prompt:
            chat += f'\n{self.name_by_role["assistant"]}:'
            
        return chat


class ChatML(ChatTemplate):
    """OpenAI's Chat Markup Language (*I think... I haven't found that much documentation online)

    Examples:
        >>> messages = [
        ...     {'role': 'system', 'content': self.system_message},
        ...     {'role': 'user', 'content': 'Hi, how are you?'},
        ...     {'role': 'assistant', 'content': 'I am doing well. How about you?'},
        ...     {'role': 'user', 'content': 'Great thanks!'},
        ...     {'role': 'assistant', 'content': 'Nice!'},
        ...     {'role': 'user', 'content': 'What is your favourite baseball team?'},
        ... ]
    
        >>> chat_template = ChatML(system_message='You are a helpful assistant.')
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

    """
    
    def __init__(self,
                 system_message: str):
        
        self.system_message = system_message
        
        
    def apply_chat_template(self,
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

    
class LlamaChatTemplate(ChatTemplate):
    """The conversation template used in LLaMA's chat fine-tuned models.

    References:
    [1] https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
    
    Examples:
        >>> messages = [
        ...     {'role': 'system', 'content': self.system_message},
        ...     {'role': 'user', 'content': 'Hi, how are you?'},
        ...     {'role': 'assistant', 'content': 'I am doing well. How about you?'},
        ...     {'role': 'user', 'content': 'Great thanks!'},
        ...     {'role': 'assistant', 'content': 'Nice!'},
        ...     {'role': 'user', 'content': 'What is your favourite baseball team?'},
        ... ]
    
        >>> chat_template = LlamaChatTemplate()
        >>> print(chat_template.apply_chat_template(messages, add_generation_prompt=True))
    
        <s>[/INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as
        possible, while being safe. Your answers should not include any harmful, unethical,
        racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses
        are socially unbiased and positive in nature.
    
        If a question does not make any sense, or is not factually coherent, explain why
        instead of answering something not correct. If you don't know the answer to a question,
        please don't share false information.
        <</SYS>>
    
        Hi, how are you? [/INST] I am doing well. How about you? </s><s>[/INST] Great
        thanks! [/INST] Nice! </s><s>[/INST] What is your favourite baseball team? [/INST]

    """
    
    DEFAULT_SYSTEM_MESSAGE = (
        'You are a helpful, respectful and honest assistant. Always answer as helpfully as '
        'possible, while being safe. Your answers should not include any harmful, unethical, '
        'racist, sexist, toxic, dangerous, or illegal content. Please ensure that your '
        'responses are socially unbiased and positive in nature.\n\nIf a question does not '
        'make any sense, or is not factually coherent, explain why instead of answering '
        'something not correct. If you don\'t know the answer to a question, please '
        'don\'t share false information.'
    )
    
    B_INST = '[INST]'
    E_INST = '[/INST]'
    B_SYS = '<<SYS>>\n'
    E_SYS = '\n<</SYS>>\n\n'
    
    def __init__(self,
                 system_message: str = DEFAULT_SYSTEM_MESSAGE):
        
        self.system_message = system_message
        

    def apply_chat_template(self,
                            messages: List[Tuple[str, str]],
                            add_generation_prompt: bool):
        # If messages[0] is a system message, merge it with first user message messages[1] 
        if messages[0]['role'] == 'system':
            messages = [
                {'role': messages[1]['role'],  # This is a user message
                 'content': self.B_SYS + messages[0]['content'] + self.E_SYS + messages[1]['content']}
            ] + messages[2:]
            
        chat = ''
        for (prompt, answer) in zip(messages[::2], messages[1::2]):
            chat += (f'<s>{self.B_INST} {(prompt["content"]).strip()} '
                     f'{self.E_INST} {(answer["content"]).strip()} </s>')

        if add_generation_prompt:
            chat += f'<s>{self.B_INST} {(messages[-1]["content"]).strip()} {self.E_INST}'

        return chat
    