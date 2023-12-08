import numpy as np
from typing import List
import unittest

import candle
from candle.nlp.chattemplates import (
    SimpleConversationTemplate,
    ChatML,
    LlamaChatTemplate,
)


class TestChatTemplates(unittest.TestCase):

    def test_message_composability(self):
        messages = [
            {'role': 'system', 'content': 'You are a super helpful model'},
            {'role': 'user', 'content': 'Hi, how are you?'},
            {'role': 'assistant', 'content': 'I am doing well. How about you?'},
            {'role': 'user', 'content': 'Great thanks!'},
            {'role': 'assistant', 'content': 'Nice!'},
            {'role': 'user', 'content': 'What is your favourite baseball team?'},
            {'role': 'assistant', 'content': 'Oreos.'},
            {'role': 'user', 'content': 'Cool!'},
        ]

        self = SimpleConversationTemplate('John', 'Bob')
        for i in range(2, 8)[::2]:
            a = self.apply_chat_template(messages[:-i], add_generation_prompt=True)
            b = self.apply_chat_template(messages, add_generation_prompt=True)
            assert b.startswith(a + ' ' + messages[-i]['content']), i

        for i in range(1, 8):
            a = self.apply_chat_template(messages[:-i], add_generation_prompt=False)
            b = self.apply_chat_template(messages, add_generation_prompt=False)
            assert b.startswith(a), i

        self = ChatML('You are a helpful assistant')
        for i in range(2, 8)[::2]:
            a = self.apply_chat_template(messages[:-i], add_generation_prompt=True)
            b = self.apply_chat_template(messages, add_generation_prompt=True)
            assert b.startswith(a + messages[-i]['content']), i

        for i in range(1, 8):
            a = self.apply_chat_template(messages[:-i], add_generation_prompt=False)
            b = self.apply_chat_template(messages, add_generation_prompt=False)
            assert b.startswith(a), i

        self = LlamaChatTemplate('You are a helpful assistant')
        for i in range(2, 8)[::2]:
            a = self.apply_chat_template(messages[:-i], add_generation_prompt=True)
            b = self.apply_chat_template(messages, add_generation_prompt=True)
            assert b.startswith(a + ' ' + messages[-i]['content']), i