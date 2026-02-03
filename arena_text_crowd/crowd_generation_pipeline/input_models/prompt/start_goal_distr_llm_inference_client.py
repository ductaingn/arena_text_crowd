from typing import Dict, List
import math
import time
import json

import os

from pydantic import BaseModel

from google import genai

from arena_simulation_setup.tree.World import WorldDescription

LLM_INSTRUCTION = """
You will be given a sentence that describes the behavior of one or more groups of humans and their interactions with the
real environment. The task is choosing the resonable and relavant positions in the map for start and goal areas for each human group trajectory base on user prompt. The world information is provided in this Arena World JSON-formated data as described below: The map is composed of a list of zones. Each zone has the following fields:
    - `name`: a unique identifier.
    - `corners`: a list of 2D points [x, y] marking the zone's 4 corners.
    - `pos`: a 2D point [x, y] marking the zone's center position.
    - `doors`: a list of doors, each defined by its name and a 2D point marking its postion [<door name>, [x, y]].
Do NOT explain anything. Output JSON only. Output must strictly follow this structure:
```json
{
"pedestrian_groups": [
    {
        "num_pedestrians": <num_pedestrians>,
        "human_models": [
            <model_1>,
            ...
            <model_n>
        ]
        "start": {
            "center_pos": [<x>,<y>],
            "size": <width, height>,
        },
        "goal": {
            "center_pos": [<x>,<y>],
            "size": <width, height>,
        },
        "group_description": <group_description>
    },
    ...
]
}
```
In which:
    - The `pedestrian_groups` field contains a list of pairs of start and goal areas of each human group, and the size of that group, each contains one `start` and one `goal`.
    - The `num_pedestrians` field contains the number of human in that group.
    - The `human_models` field contains relavant the models that later can be used to sample the human group. The available models are: ["female_adult_business_02", "female_adult_medical_01", "female_adult_police_01", "female_adult_police_02", "female_adult_police_03", "male_adult_construction_01", "male_adult_construction_02", "male_adult_construction_03", "male_adult_construction_05", "male_adult_medical_01", "male_adult_police_04"]
    - The `start` field contains informations about the chosen start area, in which:
        - `center_pos` field contains the coordinates of the center of the chosen area.
        - `size` field contains the width and height of the chosen area.
    - The `goal` field contains informations about the chosen goal area, in which, the `coners` and `location` fiels are similar to those in the `start` field.
    - The `group_description` field contains a translated sentence describing the group behavior. This field must obey these rules:
        1. Each canonicalized sentence should follow the given Structure, only contain phrases from the value part of the given
        Dictionaries, and satisfy the given pairing Constraints. The sentence should be in present tense.
        2. If a component category cannot be found in the dictionary, you need to automatically convert some nouns and verbs into
        phrases in the dictionary to ensure that the canonicalization does not exceed the scope of the dictionary. For example, the
        shape of a basketball is usually circular, so if "basketball" appears in the original description, you should use "circle" instead
        as you can only find "circle" in the dictionary.
        3. Each canonicalized sentence must start with "enter_area" action and "entrance" entity, and end with "exit_area" action and
        "exit" entity, while the sentence must contain both.
        4. If the information is insufficient, random assignments can be made based on common sense reasoning, or you can also
        make inquiries for supplementation.
        5. Some entity given in the Arena World format (e.g. `zones`, `doors`) have been converted to the supported <entity_name> (e.g. "rectangle", "zebra crossing"). And you will be given a mapping from the zones to the supported <entity_name>. You may find the relavant zones and use them as <entity_name> in the canonicalized sentences.

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
    Input: 
Generate content base given this user prompt: People run out of their room, to the hallways, and through the main hallway entrance. There should be about 5 people in each room.". Arena World <WORLD_DESCRIPTION>: 
{"zones": [{"name": "central_hallway", "corners": [[8.0, 0.0], [12.0, 0.0], [12.0, 34.2], [8.0, 34.2]], "pos": [10.0, 17.1], "doors": [["main_hallway_entrance", [10.0, 0.0]]]}, {"name": "reception", "corners": [[0.0, 0.0], [8.0, 0.0], [8.0, 5.0], [0.0, 5.0]], "pos": [4.0, 2.5], "doors": [["reception_door", [8.0, 2.5]]]}, {"name": "exam_room_1", "corners": [[0.0, 5.05], [8.0, 5.05], [8.0, 10.05], [0.0, 10.05]], "pos": [4.0, 7.550000000000001], "doors": [["exam_room_1_door", [8.0, 7.550000000000001]]]}, {"name": "patient_ward", "corners": [[0.0, 10.1], [8.0, 10.1], [8.0, 25.1], [0.0, 25.1]], "pos": [4.0, 17.6], "doors": [["patient_ward_door_1", [8.0, 14.1]], ["patient_ward_door_2", [8.0, 21.1]]]}, {"name": "pharmacy", "corners": [[0.0, 25.15], [8.0, 25.15], [8.0, 30.15], [0.0, 30.15]], "pos": [4.0, 27.65], "doors": [["pharmacy_door", [8.0, 27.65]]]}, {"name": "nurse_resting_room", "corners": [[12.0, 0.0], [18.5, 0.0], [18.5, 10.0], [12.0, 10.0]], "pos": [15.25, 5.0], "doors": [["nurse_resting_door", [15.0, 10.0]]]}, {"name": "waiting_area", "corners": [[18.5, 0.0], [25.0, 0.0], [25.0, 10.0], [18.5, 10.0]], "pos": [21.75, 5.0], "doors": [["waiting_to_sub_2", [22.0, 10.0]]]}, {"name": "sub_hallway", "corners": [[12.0, 10.05], [25.0, 10.05], [25.0, 13.05], [12.0, 13.05]], "pos": [18.5, 11.55], "doors": []}, {"name": "operating_room", "corners": [[12.0, 13.1], [18.0, 13.1], [18.0, 21.6], [12.0, 21.6]], "pos": [15.0, 17.35], "doors": [["or_door", [12.0, 17.35]]]}, {"name": "laboratory", "corners": [[12.0, 21.65], [18.0, 21.65], [18.0, 30.15], [12.0, 30.15]], "pos": [15.0, 25.9], "doors": [["laboratory_door", [12.0, 25.9]]]}, {"name": "wc_pharmacy", "corners": [[0.0, 30.2], [8.0, 30.2], [8.0, 34.2], [0.0, 34.2]], "pos": [4.0, 32.2], "doors": [["wc_pharmacy_door", [8.0, 32.25]], ["wc_stall_1_door", [2.0, 30.85]], ["wc_stall_2_door", [2.0, 32.15]], ["wc_stall_3_door", [2.0, 33.45]]]}, {"name": "wc_waiting", "corners": [[18.05, 13.1], [22.05, 13.1], [22.05, 17.1], [18.05, 17.1]], "pos": [20.05, 15.100000000000001], "doors": [["wc_waiting_door", [20.0, 13.1]], ["wc_waiting_stall_1_door", [20.05, 13.75]], ["wc_waiting_stall_2_door", [20.05, 15.08]], ["wc_waiting_stall_3_door", [20.05, 16.42]]]}, {"name": "doctors_resting_room", "corners": [[12.0, 30.2], [18.0, 30.2], [18.0, 34.2], [12.0, 34.2]], "pos": [15.0, 32.2], "doors": [["doctors_resting_room_door", [12.0, 32.25]]]}]}. Arena regions to Semantic entity mapping:
Arena World Entity | Supported Semantic Entity
main_hallway_entrance | passage
central_hallway | zebra_crossing
reception_door | passage
exam_room_1_door | passage
patient_ward_door_1 | passage
patient_ward_door_2 | passage
pharmacy_door | passage
nurse_resting_door | passage
waiting_to_sub_2 | passage
sub_hallway | zebra_crossing
or_door | passage
laboratory_door | passage
wc_pharmacy_door | passage
wc_stall_1_door | passage
wc_stall_2_door | passage
wc_stall_3_door | passage
wc_waiting_door | passage
wc_waiting_stall_1_door | passage
wc_waiting_stall_2_door | passage
wc_waiting_stall_3_door | passage
doctors_resting_room_door | passage

    Output:
```json
{
"pedestrian_groups": [
    {
        "num_pedestrians": 4,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01"
        ]
        "start": {
            "center_pos": [4.0, 2.5],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the bottom most left entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
"
    },
    {
        "num_pedestrians": 5,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01"
        ]
        "start": {
            "center_pos": [4.0, 7.55],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the second up from bottom, most left entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 6,
        "human_models": [
            "male_adult_construction_03", "male_adult_construction_05", "male_adult_medical_01"
        ]
        "start": {
            "center_pos": [4.0, 17.6],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the middle most left entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 4,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01", "male_adult_medical_01"
        ]
        "start": {
            "center_pos": [4.0, 27.65],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the second from top, most left entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 5,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01", "male_adult_medical_01"
        ]
        "start": {
            "center_pos": [15.25, 5.0],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the bottom middle entry, walks throught the nearby passage, moves pass the right zebra crossing, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 4,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01", "male_adult_medical_01"
        ]
        "start": {
            "center_pos": [15.0, 17.35],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the second from bottom right entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 4,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01"
        ]
        "start": {
            "center_pos": [15.0, 25.9],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the second from top right entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 4,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01", "male_adult_medical_01"
        ]
        "start": {
            "center_pos": [15.0, 32.2],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A small group moves from the top most right entry, walks throught the nearby passage, passes by the left zebra crossing, exits through the bottom exit."
    },
    {
        "num_pedestrians": 6,
        "human_models": [
            "female_adult_business_02", "female_adult_medical_01", "female_adult_police_01", "female_adult_police_02", "female_adult_police_03", "male_adult_construction_01", "male_adult_construction_02", "male_adult_construction_03", "male_adult_construction_05", "male_adult_medical_01", "male_adult_police_04"
        ]
        "start": {
            "center_pos": [21.75, 5.0],
            "size": [1.5, 1.5],
        },
        "goal": {
            "center_pos": [10.0, 1.0],
            "size": [1.5, 1.5],
        },
        "group_description": "A large group moves from the bottom most right entry, walks throught the nearby passage, moves pass the right zebra crossing, passes by the left zebra crossing, exits through the bottom exit."
    },
]
}
```
"""


class Area(BaseModel):
    center_pos: List[float]
    size: List[float]


class PedestrianGroup(BaseModel):
    num_pedestrians: int
    human_models: List[str]
    start: Area
    goal: Area
    group_description: str


class LLMResponse(BaseModel):
    pedestrian_groups: List[PedestrianGroup]


class StartGoalDistrLLMInferenceClient:
    """
    Use Large Language Model to predict the start and goal distribution
    """

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
            response_mime_type="application/json",
            top_p=self.top_p,
            thinking_config=genai.types.ThinkingConfig(
                include_thoughts=False, thinking_budget=self.thinking_budget
            ),
        )

    def preprocess_world_description(self, world_description: WorldDescription) -> str:
        """
        Preprocesses the world description, keeps corners and walls only and converts them to 2D format.

        Args:
            world_description : WorldDescription
                The world description to preprocess.

        Returns:
            parsed : str
                The preprocessed JSON formatted str world description.
        """
        parsed = {}

        parsed["zones"] = []
        for zone in world_description.zones:
            parsed_zone = {
                "name": zone.name,
                "corners": [[corner.x, corner.y] for corner in zone.corners],
                "pos": [zone.floor.pos.x, zone.floor.pos.y],
                "doors": [
                    [
                        door.name,
                        [
                            (door.start.x + door.end.x) / 2,
                            (door.start.y + door.end.y) / 2,
                        ],
                    ]
                    for door in zone.doors
                ],
            }
            parsed["zones"].append(parsed_zone)

        return json.dumps(parsed)

    def inference(
        self,
        user_prompt: str,
        arena_world_desc: WorldDescription,
        arena_entity_to_semantic_entity_map: Dict[str, str],
    ) -> LLMResponse:
        """
        Select the start and goal areas, canonicalize prompts given Arena WorldDescription according to user prompt

        Returns:
            start_goal_areas: Dict[str, str]
                Start and goal pair for each group.
        """
        world_info = self.preprocess_world_description(arena_world_desc)
        parsed_mapping = "Arena World Entity | Supported Semantic Entity"
        for arena_entity, semantic_type in arena_entity_to_semantic_entity_map.items():
            parsed_mapping += f"\n{arena_entity} | {semantic_type}"
        print("Start inference start and goal zones...")
        start = time.time()
        messages = []
        messages.append(
            f"Generate content base given this user prompt: {user_prompt}. Arena World <WORLD_DESCRIPTION>: {world_info}. Arena regions to Semantic entity mapping: {arena_entity_to_semantic_entity_map}"
        )
        response = self.inference_client.models.generate_content(
            model=self.model, contents=messages, config=self.generate_content_config
        )
        end = time.time()
        print(f"Inference done, took: {end - start:.1f}s")

        answer = response.text
        assert answer is not None

        if answer.startswith("```json"):
            answer = answer.strip("```json").strip("```").strip()
        elif answer.startswith("```"):
            answer = answer.strip("```").strip()

        try:
            pedestrian_groups = LLMResponse.model_validate_json(answer)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM response: {e}")
            print("Returning empty start and goal zones!")
            pedestrian_groups = LLMResponse(pedestrian_groups=[])

        return pedestrian_groups
