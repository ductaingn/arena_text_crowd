import numpy as np
import csv

from arena_simulation_setup.tree.World import WorldDescription
from arena_text_crowd.converters.arena_world_to_text_crowd_scenario import get_arena_world_size
from arena_text_crowd.crowd_generation_pipeline.input_models.scenario import Scenario
from arena_text_crowd.crowd_generation_pipeline.crowd_generation_pipeline import PipelineOutput

def text_crowd_output_to_socnavbench_scenario(
    pipeline_output: PipelineOutput, 
    output_csv_path: str, 
    text_crowd_scenario: Scenario, 
    arena_world: WorldDescription
):
    """
    Convert the generated agent trajectories from Text-Crowd to a SocNavBench CSV format.
    The output CSV contains 4 rows: Frame IDs, Agent IDs, X-coordinates, Y-coordinates.
    """
    # 1. Calculate scaling factors between Text-Crowd scenario and Arena World
    arena_world_size = get_arena_world_size(arena_world)
    text_crowd_scenario_size = text_crowd_scenario.scenario_config.window_size
    
    scale_x = arena_world_size[0] / text_crowd_scenario_size[0]
    scale_y = arena_world_size[1] / text_crowd_scenario_size[1]

    # 2. Collect all trajectory points
    # SocNavBench format requires points to be ordered by frame, then by agent ID.
    all_samples = []
    
    for group in pipeline_output.groups:
        for agent in group.agent_list:
            # agent_trajectory is assumed to be a list of (x, y) tuples
            for frame_idx, (x, y) in enumerate(agent.agent_trajectory):
                all_samples.append({
                    'frame': frame_idx+1,
                    'id': agent.agent_id,
                    'x': x * scale_x,
                    'y': y * scale_y
                })

    # 3. Sort samples by Frame ID, then by Agent ID (to match interleaving in default.csv)
    all_samples.sort(key=lambda s: (s['frame'], s['id']))

    # 4. Prepare the four attribute rows
    row_frames = [s['frame'] for s in all_samples]
    row_ids = [s['id'] for s in all_samples]
    row_xs = [round(s['x'], 4) for s in all_samples]
    row_ys = [round(s['y'], 4) for s in all_samples]

    # 5. Write to CSV
    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_frames)
        writer.writerow(row_ids)
        writer.writerow(row_xs)
        writer.writerow(row_ys)

    print(f"Successfully converted trajectories to {output_csv_path}")
    scenario_duration = max(row_frames)*(1/25.0)  # Assuming 25 FPS
    print(f"Total frames: {max(row_frames)}, Total agents: {len(set(row_ids))}, Scenario duration: {scenario_duration:.2f}s")