from typing import Dict, List
import time
import random

import os

from google import genai


LLM_INSTRUCTION = """
Instruction:
You will be given a sentence that describes the behavior of one or more groups of humans and their interactions with the
real environment. The task is translating the given sentence into several canonicalized sentences. Specifically:
1. The given sentence should be divided into multiple canonicalized sentences, where each sentence only describes the
behavior of one human group.
2. Each canonicalized sentence should follow the given Structure, only contain phrases from the value part of the given
Dictionaries, and satisfy the given pairing Constraints. The sentence should be in present tense.
3. If a component category cannot be found in the dictionary, you need to automatically convert some nouns and verbs into
phrases in the dictionary to ensure that the canonicalization does not exceed the scope of the dictionary. For example, the
shape of a basketball is usually circular, so if "basketball" appears in the original description, you should use "circle" instead
as you can only find "circle" in the dictionary.
4. Each canonicalized sentence must start with "enter_area" action and "entrance" entity, and end with "exit_area" action and
"exit" entity, while the sentence must contain both.
5. If the information is insufficient, random assignments can be made based on common sense reasoning, or you can also
make inquiries for supplementation.

Structure of the canonicalized sentence:
A <group_size> group
<action> the( <action_location> side of the)( <entity_location>) <entity_name>( [action_adjective]),
<action> the( <action_location> side of the)( <entity_location>) <entity_name>( [action_adjective]),
...
There are two optional substructures for [action_adjective]:
1. from <action_location> to <action_location>
2. <action_direction>

The elements in angle brackets represent the sentence components that need to be filled in with phrases from the
dictionary. The content in parentheses is allowed to be dropped out if it is difficult to fill.

Dictionaries:
Each type of sentence component has a specified dictionary that contains several key-value list pairs, which is formatted as:
<sentence component>:
{"key": ["value_1", "value_2", ...], ...}
The "key" is a representation of one component category and the corresponding ["value_1", "value_2", ...] is a set of available
phrases of this component category. Only values can be used in filling the canonicalized sentence

<action>:
{"enter_area": ["enters from", "gets in from", "moves from"], "exit_area": ["exits through", "leaves through", "quits through"],
"pass_by": ["passes", "passes by", "moves past"], "pass_by_edge": ["passes", "passes by", "moves past"], "around": ["bypasses",
"circles around"], "cross": ["crosses", "passes", "walks across", "moves across"], "through": ["moves through", "walks through",
"passes through", "travels through"]}

<action_location> and <entity_location>:
{"right": ["right"], "upper_right": ["upper right"], "upper": ["upper"], "upper_left": ["upper left"], "left": ["left"], "lower_left":
["lower left"], "lower": ["lower"], "lower_right": ["lower right"], "top_right": ["top right"], "top": ["top"], "top_left": ["top left"],
"bottom_left": ["bottom left"], "bottom": ["bottom"], "bottom_right": ["bottom right"], "middle": ["middle", "center"]}

<entity_name>:
{"rectangle": ["rectangle", "oblong", "quadrilateral"], "triangle": ["triangle", "triangular shape"], "circle": ["circle", "ring",
"round"], "zebra_crossing": ["zebra crossing", "pedestrian crossing", "crosswalk"], "passage": ["passage", "pathway", "corridor"],
"entrance": ["entrance", "entry"], "exit": ["exit", "export"]}

<action_direction>:
{"anticlockwise": ["anticlockwise", "counterclockwise"], "clockwise": ["clockwise"]}

<group_size>:
{"tiny": ["tiny"], "small": ["small"], "big": ["big"], "large": ["large"]}

Constraints:
In each clause of the canonicalized sentence, each category of <entity_name> can only be paired with a constrained category
list of <action>. Below are the <action> categories supported by each <entity_name>:
"entrance": "enter_area"
"exit": "exit_area"
"rectangle", "triangle", "circle": "pass_by", "pass_by_edge", "around"
"zebra_crossing": "pass_by", "cross"
"passage": "pass_by", "through"

Example:
Given:
I want two groups where one group is moving from left to right and cross a passage in this process and another group enters
from the bottom right entry and then walks around the upper circle, and finally leaves through the upper left.
Return:
A tiny group moves from the left entry, crosses the passage, exits through the right exit.
A large group enters from the bottom right entrance, circles around the upper circle, leaves through the upper left export.

Given:
There are two huge groups of humans located at the upper right corner of the map and the upper left corner respectively and
trying to cross the middle narrow passage and escape from the exit below the map.
Return:
A large group moves from the upper right entrance, crosses the middle passage, exits through the bottom exit.
A large group moves from the upper left entrance, crosses the middle passage, exits through the bottom exit.
"""


class PromptCanonicalizer:
    def __init__(self, model="gemini-2.5-flash", top_p=0.8, thinking_budget=8192):
        self.model = model
        self.top_p = top_p
        self.thinking_budget = thinking_budget

        if "GEMINI_API_KEY" not in os.environ:
            print("GEMINI_API_KEY environment variable not set!")
            raise OSError("GEMINI_API_KEY environment variable not set!")

        self.inference_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        self.generate_content_config = genai.types.GenerateContentConfig(
            system_instruction=LLM_INSTRUCTION,
            top_p=self.top_p,
            thinking_config=genai.types.ThinkingConfig(
                include_thoughts=False, thinking_budget=self.thinking_budget
            ),
        )

    def canonicalize(self, user_prompt: str) -> List[str]:
        print("Canonicalizing prompt ...")
        start = time.time()
        messages = [user_prompt]
        response = self.inference_client.models.generate_content(
            model=self.model, contents=messages, config=self.generate_content_config
        )
        end = time.time()
        answer = response.text
        assert answer is not None

        answer = answer.splitlines()

        print(f"Canonicalizing done, took: {end - start:.1f}s")

        return answer

    def get_group_size(self, canonicalized_des: str) -> List[int]:
        # TODO: process this
        group_n = len(canonicalized_des)
        group_size = [random.randint(1, 10)] * group_n

        return group_size
