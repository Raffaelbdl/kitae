from rl.algos.pipelines.experience_pipeline import ExperienceTransform
from rl.algos.pipelines.update_pipeline import UpdateModule


class IntrinsicRewardModule(ExperienceTransform, UpdateModule): ...
