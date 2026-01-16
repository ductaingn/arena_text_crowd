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
real environment. The task is choosing the resonable start and goal zones for each human group trajectory. The world information is provided in this JSON-formated data as described below: The map is composed of a list of zones. Each zone has the following fields:
    - `name`: a unique identifier.
    - `corners`: a list of 2D points [x, y] marking the zone's corners, you can calculate the zone's position and coverage, and check if a point is within a zone or not base on these points.
    - `walls`: a list of wall segments, each defined by two 2D points [[x1, y1], [x2, y2]].
    - `mat`: the material of the floor (can be empty).
    - `entities`: contains static objects in the zone. Each static object has:
    -   - `name`: the object's unique name.
    -   - `model`: the type of object (e.g., `shelf`).
    -   - `pose`: a list [x, y, yaw] representing the object's position and rotation.
    - `description`: a human-readable name of the zone.
Do NOT explain anything. Output JSON only. Output must strictly follow this structure:
```json
{
"start_goal_zones": [
    {
        "start": {
            "name": <zone_name>,
            "location": <zone_location_description>
        },
        "goal": {
            "name": <zone_name>,
            "location": <zone_location_description>
        },
    },
    ...
]
}
```
In which:
    - The `start_goal_zones` field contains a list of pairs of start and goal zones of each human group, each pair contains one `start` and one `goal`.
    - The `start` field contains informations about the chosen zone, in which:
        - `name` field is the exact name of the chosen zone given in the world description.
        - `location` field is the description of where the zone is in the world, in compare to the other zones.
    - The `goal` field contains informations about the chosen zone, in which:
        - `name` field is the exact name of the chosen zone given in the world description.
        - `location` field is the description of where the zone is in the world, in compare to the other zones.
    - The `location` value can only be chosen from this dictionary: ["right, "upper right", "upper", "upper left", "left", "lower left", "lower", "lower right", "top right", "top", "top left", "bottom left", "bottom", "bottom right", "middle", "center"] or something similar.
"""


class Zone(BaseModel):
    name: str
    location: str


class StartGoalPair(BaseModel):
    start: Zone
    goal: Zone


class LLMResponse(BaseModel):
    start_goal_zones: List[StartGoalPair]


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
                "walls": [
                    [[wall.start.x, wall.start.y], [wall.end.x, wall.end.y]]
                    for wall in zone.walls
                ],
                "entities": [
                    {
                        "name": entity.name,
                        "model": entity.model.serialize(),
                        "pose": [
                            entity.pose.position.x,
                            entity.pose.position.y,
                            math.degrees(
                                entity.pose.orientation.to_yaw()
                            ),  # I use degree for yaw for now (look at `context.py``)
                        ],
                    }
                    for entity in zone.entities.static
                ],
            }
            parsed["zones"].append(parsed_zone)

        return json.dumps(parsed)

    def inference(
        self, user_prompt: str, arena_world_desc: WorldDescription
    ) -> LLMResponse:
        """
        Select the Zones given Arena WorldDescription according to user prompt

        Returns:
            start_goal_zones: Dict[str, str]
                Start and goal pair for each group.
        """
        world_info = self.preprocess_world_description(arena_world_desc)
        print("Start inference start and goal zones...")
        start = time.time()
        messages = []
        messages.append(
            f"Chose start and goal zones with this user prompt: {user_prompt}. Generate data base on this world data as below <WORLD_DESCRIPTION>: {world_info}."
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
            start_goal_zones = LLMResponse.model_validate_json(answer)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM response: {e}")
            print("Returning empty start and goal zones!")
            start_goal_zones = LLMResponse(start_goal_zones=[])

        return start_goal_zones
