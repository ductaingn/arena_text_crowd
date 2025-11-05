# Text-Crowd adapter for Arena-RosNav
This code is mostly borrowed and modified from https://github.com/MLZG/Text-Crowd

Example usage
```python
from arena_text_crowd.crowd_generation_pipeline import (
    CrowdGenerationPipeline, 
    CrowdGenerationPipelineConfig
)

from arena_text_crowd.converters import (
    arena_world_to_crowd_text_semantic_map,
    crowd_text_output_to_socnavbench_scenario
)

from arena_simulation_setup. ... import WorldDescription

crowd_generator = CrowdGenerationPipeline(
    CrowdGenerationPipelineConfig(...),
    ...
)

user_prompt = """
...
"""

arena_world = WorldDescription()

crowd_text_output = crowd_generator.generate(
    arena_world_to_crowd_text_semantic_map(arena_world),
    user_prompt
)

scenario = crowd_text_output_to_socnavbench_scenario(crowd_text_output)
```
