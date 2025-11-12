# Text-Crowd adapter for Arena-RosNav
This code is mostly borrowed and modified from https://github.com/MLZG/Text-Crowd

In the origin work, it seems like the number of agent for each group is not determined by the LLM and user prompt, but randomized from the  authors' defined ranges `group_size_ranges={"tiny":[2, 3], "small":[4, 7], "big":[8, 15], "large":[16, 30]}`, and they used the generated training label (ground truth group size ranges) for the inference/validation part (Trace for `group_sizes` in https://github.com/MLZG/Text-Crowd/blob/master/Language_Crowd_Animation/Quantitative_Exps.py/#L395). In this implementation, we also use the same defined `group_size_ranges`, but the number of agents for each group will be randomized in the runtime. Similar thing also happened to the path of each group.


## Example usage
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
