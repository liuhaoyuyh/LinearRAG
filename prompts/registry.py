from .loader import PromptLoader


class PromptAccessor:
    """
    Represents a single prompt entry, providing rendering and metadata helpers.
    """

    def __init__(self, namespace: str, name: str):
        self.path = f"{namespace}/{name}"

    def _prompt(self):
        return PromptLoader.load(self.path)

    def __call__(self, **kwargs):
        return self.render(**kwargs)

    def render(self, **kwargs):
        prompt = self._prompt()
        return prompt.render(**kwargs)

    def get_template(self) -> str:
        return self._prompt().template

    def get_id(self) -> str:
        return self._prompt().id

    def get_version(self) -> str:
        return self._prompt().version

    def get_description(self) -> str:
        return self._prompt().description

    def get_metadata(self) -> dict:
        prompt = self._prompt()
        return {
            "id": prompt.id,
            "version": prompt.version,
            "description": prompt.description,
        }


# ===========================
# Base API for a namespace
# ===========================
class PromptNamespace:
    def __init__(self, namespace: str):
        self.namespace = namespace

    def _accessor(self, name: str) -> PromptAccessor:
        return PromptAccessor(self.namespace, name)


# ===========================
# video_planner
# ===========================
class VideoPlannerPrompts(PromptNamespace):
    def __init__(self):
        super().__init__("video_planner")
        self.context_learning_animation_narration = self._accessor(
            "context_learning_animation_narration"
        )
        self.context_learning_code = self._accessor("context_learning_code")
        self.context_learning_scene_plan = self._accessor(
            "context_learning_scene_plan"
        )
        self.context_learning_technical_implementation = self._accessor(
            "context_learning_technical_implementation"
        )
        self.context_learning_vision_storyboard = self._accessor(
            "context_learning_vision_storyboard"
        )
        # -------- Scene plan 系列 --------
        self.scene_animation_narration = self._accessor("scene_animation_narration")
        self.scene_plan = self._accessor("scene_plan")
        self.scene_technical_implementation = self._accessor(
            "scene_technical_implementation"
        )
        self.scene_vision_storyboard = self._accessor("scene_vision_storyboard")

class MindMapPrompts(PromptNamespace):
    def __init__(self):
        super().__init__("mind_map")
        self.abstract_prompts = self._accessor("abstract_prompts")
        self.experiments_result_prompts = self._accessor(
            "experiments_result_prompts"
        )
        self.introduction_prompts = self._accessor(
            "introduction_prompts"
        )
        self.method_prompts = self._accessor(
            "method_prompts"
        )
        self.related_prompts = self._accessor(
            "related_prompts"
        )
        self.title_prompts = self._accessor(
            "title_prompts"
        )
        self.system_prompts = self._accessor(
            "system_prompts"
        )

class qaPrompts(PromptNamespace):
    def __init__(self):
        super().__init__("qa")
    pass



# ===========================
# template（如果你有 template prompts）
# ===========================
class TemplatePrompts(PromptNamespace):
    def __init__(self):
        super().__init__("template")

    # 示例
    @property
    def basic(self) -> PromptAccessor:
        return self._accessor("basic")


# ===========================
# tasks
# ===========================
class TaskPrompts(PromptNamespace):
    def __init__(self):
        super().__init__("tasks")

    # 示例
    @property
    def something(self) -> PromptAccessor:
        return self._accessor("something")


# ===========================
# 总注册对象（顶层可补全）
# ===========================
class PromptsAPI:
    def __init__(self):
        self.video_planner = VideoPlannerPrompts()
        self.mind_map = MindMapPrompts()



prompts = PromptsAPI()
