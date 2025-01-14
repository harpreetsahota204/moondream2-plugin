import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import requests 

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from moondream import (
        run_moondream_model
    )

def _handle_calling(
        uri, 
        sample_collection, 
        revision,
        operation,
        output_field,
        delegate=False
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_name=revision,
        operation=operation,
        output_field=output_field,
        delegate=delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

MOONDREAM_MODES = {
    "caption": "Caption images", 
    "query": "Visual question answering",
    "detect": "Object detection",
    "point": "Apply point on object",
}


# Load Moondream2 revisions from HuggingFace
MOONDREAM_VERSIONS_URL = "https://huggingface.co/vikhyatk/moondream2/raw/main/versions.txt"

def load_moondream_versions():
    try:
        response = requests.get(MOONDREAM_VERSIONS_URL)
        response.raise_for_status()
        # Split by newlines and remove any empty strings
        versions = [v.strip() for v in response.text.splitlines() if v.strip()]
        return versions
    except Exception as e:
        print(f"Failed to load Moondream versions: {e}")
        return ["2025-01-09"]  # Return default version as fallback

# Load available versions
MOONDREAM_REVISIONS = load_moondream_versions()

class MoondreamOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="moondream",
            label="Run Moondream2",
            description="Run the Moondream model on your Dataset!",
            dynamic=True,
            icon="/assets/moon-phase-svgrepo-com.svg",
            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        mode_dropdown = types.Dropdown(label="What would you like to use Moondream2 for?")
        
        for arch_key, arch_value in MOONDREAM_MODES.items():
            mode_dropdown.add_choice(arch_value, label=arch_key)

        revision_dropdown = types.Dropdown(label="Which revision would you like to use?")

        for revision in MOONDREAM_REVISIONS:  # Add the available revisions
            revision_dropdown.add_choice(revision, label=revision)

        inputs.enum(
            "revision",
            values=revision_dropdown.values(),
            label="Revision",
            default="2025-01-09",
            description="Select from one of the available revisions. Note: The model weights will be downloaded from Hugging Face.",
            view=revision_dropdown,
            required=False
        )

        inputs.enum(
            "operation",
            values=mode_dropdown.values(),
            label="Moondream2 Tasks",
            description="Select from one of the supported tasks.",
            view=mode_dropdown,
            required=True
        )

        length_radio = types.RadioGroup()
        length_radio.add_choice("short", label="A short caption")
        length_radio.add_choice("normal", label="A more descriptive caption")        
        
        chosen_task = ctx.params.get("operation")

        if chosen_task == "caption":
            inputs.enum(
                "length",
                label="Caption Length",
                description="Which caption type would you like?",
                required=True,
                view=length_radio
            )

        if chosen_task == "query":
            inputs.str(
                "query_text",
                label="Query",
                description="What's your query?",
                required=True,
            )

        if chosen_task == "detect":
            inputs.str(
                "object_type",
                label="Detect",
                description="What do you want to detect? Currently this model only supports passing one object.",
                required=True,
            )

        if chosen_task == "point":
            inputs.str(
                "object_type",
                label="Point",
                description="What do you want to place a point on? Currently this model only supports passing one object",
                required=True,
            )
       

        inputs.str(
            "output_field",            
            required=True,
            label="Output Field",
            description="Name of the field to store the results in."
            )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        revision = ctx.params.get("revision")
        operation = ctx.params.get("operation")
        output_field = ctx.params.get("output_field")
        
        # Create kwargs dictionary with additional parameters based on operation
        kwargs = {}
        if operation == "caption":
            kwargs["length"] = ctx.params.get("length")
        elif operation == "query":
            kwargs["query_text"] = ctx.params.get("query_text")
        elif operation in ["detect", "point"]:
            kwargs["object_type"] = ctx.params.get("object_type")
      
        run_moondream_model(
            dataset=view,
            revision=revision,
            operation=operation,
            output_field=output_field,
            **kwargs
            )
        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            revision,
            operation,
            output_field,
            delegate,
            **kwargs
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            revision,
            operation,
            output_field,
            delegate,
            **kwargs
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(MoondreamOperator)
