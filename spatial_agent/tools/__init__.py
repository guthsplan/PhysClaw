"""ToolsModule: injected into the Jupyter kernel as ``tools``.

Usage in kernel::

    window = tools.Video.window(center_time=3.0, radius=0.5, num_frames=16)
    track = tools.SAM3.track("car", window)
    recon = tools.Reconstruct.run(window)
    show(track.overlay(window[0]))
    pt = tools.TAP.track(track.centroid(window[0]), window)
    fit = tools.Motion.fit(pt.xy_with_time(), degree=2)
    tools.Geometry.norm(fit.velocity(3.0))
    chart = tools.Graph.plot(distances, x_label="Frame", y_label="Dist (m)")

Transport-level methods (``_segment_video_by_text`` etc.) are private and
are not injected into the kernel.

Perception/tracking tools (Reconstruct, SAM3, TAP) are only constructed and
only advertised in the system prompt when listed in ``config.tools_to_use``
-- this keeps single-tool ablations (e.g. TAP-only) clean: an unlisted tool
is not just undocumented, it does not exist as a ``tools.X`` attribute.
"""

from spatial_agent.tools.mask_utils import MaskUtils
from spatial_agent.tools.geometry_utils import GeometryUtils
from spatial_agent.tools.graph_drawer import GraphDrawer
from spatial_agent.tools.draw_utils import DrawUtils
from spatial_agent.tools.video_utils import VideoUtils
from spatial_agent.tools.motion_utils import MotionUtils


class ToolsModule:
    """Assembled tool namespace injected as ``tools`` in the kernel."""

    TOOL_NAMES = ("Reconstruct", "SAM3", "TAP", "Graph", "Video", "Mask", "Geometry", "Motion", "Draw")

    def __init__(
        self,
        config,
        metadata,
        input_images=None,
        input_images_list=None,
        video_sources_list=None,
    ):
        tools_to_use = set(getattr(config, "tools_to_use", None) or [])
        gpu_retries = getattr(config, "gpu_tool_max_retries", 3)

        self.SAM3 = None
        self.Reconstruct = None
        self.TAP = None

        if "SAM3" in tools_to_use or "Reconstruct" in tools_to_use:
            from spatial_agent.tools.sam3_tool import SAM3Tool
            # SAM3 must be created before Reconstruct so the instance is available
            self.SAM3 = SAM3Tool(
                video_source=getattr(metadata, "video_source", None),
                input_images=input_images,
                gpu_tool_max_retries=gpu_retries,
                sam3_max_video_frames=getattr(config, "sam3_max_video_frames", 200),
                total_video_frames=metadata.total_frames or 0,
                num_videos=getattr(metadata, "num_videos", 1),
                input_images_list=input_images_list,
                video_sources_list=video_sources_list,
            )
        if "Reconstruct" in tools_to_use:
            from spatial_agent.tools.reconstruct_tool import ReconstructTool
            self.Reconstruct = ReconstructTool(
                self.SAM3, config,
                gpu_tool_max_retries=gpu_retries,
                metadata=metadata,
            )
        if "TAP" in tools_to_use:
            from spatial_agent.tools.tap_tool import TAPTool
            self.TAP = TAPTool(fps=metadata.fps or 1.0)

        # CPU tools (direct, in-process) -- always available
        self.Graph = GraphDrawer()
        self.Video = VideoUtils(
            fps=metadata.fps or 1.0,
            total_frames=metadata.total_frames or 0,
            input_images=input_images,
            input_images_list=input_images_list,
        )
        self.Mask = MaskUtils()
        self.Geometry = GeometryUtils()
        self.Motion = MotionUtils()
        self.Draw = DrawUtils()

    def get_all_prompt_descriptions(self, **format_kwargs) -> str:
        """Aggregate TOOL_PROMPT_DESCRIPTION from all instantiated tools.

        Args:
            **format_kwargs: Passed to str.format() on each description
                (e.g. reconstruct_max_frames=32).

        Returns:
            Concatenated tool descriptions ready for the system prompt.
        """
        from spatial_agent.config import get_config
        ablations = get_config().prompt_section_ablations

        tools = [
            t for t in (self.Reconstruct, self.SAM3, self.TAP)
            if t is not None
        ] + [self.Graph, self.Video, self.Mask, self.Geometry, self.Motion, self.Draw]
        parts = []
        for tool in tools:
            desc = tool.get_prompt_description(ablations=ablations)
            if desc:
                try:
                    desc = desc.format(**format_kwargs)
                except KeyError:
                    pass  # leave unformatted placeholders as-is
                parts.append(desc)
        return "\n".join(parts)

    @staticmethod
    def get_all_prompt_descriptions_static(**format_kwargs) -> str:
        """Aggregate descriptions from all tool classes (no instances needed).

        Only includes perception/tracking tools (Reconstruct, SAM3, TAP) that
        are listed in ``tools_to_use`` config. CPU utility tools are always
        included. Ablation config is passed through so tool sub-sections can
        be individually excluded or overridden.
        """
        from spatial_agent.config import get_config
        from spatial_agent.tools.reconstruct_tool import ReconstructTool
        from spatial_agent.tools.sam3_tool import SAM3Tool
        from spatial_agent.tools.tap_tool import TAPTool

        config = get_config()
        tools_to_use = set(config.tools_to_use or [])
        ablations = config.prompt_section_ablations

        # Perception/tracking tools: only include if configured
        gated_tool_map = {
            "Reconstruct": ReconstructTool,
            "SAM3": SAM3Tool,
            "TAP": TAPTool,
        }
        # CPU utility tools: always included
        cpu_classes = [GraphDrawer, VideoUtils, MaskUtils, GeometryUtils, MotionUtils, DrawUtils]

        parts = []
        for name, cls in gated_tool_map.items():
            if name in tools_to_use:
                desc = cls.get_prompt_description(ablations=ablations)
                if desc:
                    try:
                        desc = desc.format(**format_kwargs)
                    except KeyError:
                        pass
                    parts.append(desc)

        for cls in cpu_classes:
            desc = cls.get_prompt_description(ablations=ablations)
            if desc:
                try:
                    desc = desc.format(**format_kwargs)
                except KeyError:
                    pass
                parts.append(desc)

        return "\n".join(parts)

    @staticmethod
    def get_all_tool_section_names() -> set:
        """Collect all valid tool ablation section names for typo warnings."""
        from spatial_agent.tools.base import get_all_tool_ablation_names
        from spatial_agent.tools.reconstruct_tool import ReconstructTool
        from spatial_agent.tools.sam3_tool import SAM3Tool
        from spatial_agent.tools.tap_tool import TAPTool

        return get_all_tool_ablation_names(
            ReconstructTool, SAM3Tool, TAPTool,
            GraphDrawer, VideoUtils, MaskUtils, GeometryUtils, MotionUtils, DrawUtils,
        )

    def __repr__(self) -> str:
        tools = self.TOOL_NAMES
        return f"ToolsModule(available={tools})"
