from rl_tools.algos.pipelines.experience_pipeline import ExperienceTransform
from rl_tools.algos.pipelines.update_pipeline import UpdateModule


class PipelineModule(ExperienceTransform, UpdateModule):
    @property
    def experience_transform(self) -> ExperienceTransform:
        return ExperienceTransform(
            process_experience_fn=self.process_experience_fn, state=self.state
        )

    @property
    def update_module(self) -> UpdateModule:
        return UpdateModule(update_fn=self.update_fn, state=self.state)

