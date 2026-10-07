"""Optional ADIOS2 input, converted to [variable, section, time, height, width]."""

import numpy as np


def canonicalize(data, axes):
    """Reorder named axes and insert missing variable/section singleton axes."""
    target = ["variable", "section", "time", "height", "width"]
    axes = list(axes)
    if (len(axes) != data.ndim or len(set(axes)) != len(axes)
            or not set(axes).issubset(target)
            or not {"time", "height", "width"}.issubset(axes)
            or not 3 <= data.ndim <= 5):
        raise ValueError(
            f"Invalid axes {axes} for shape {data.shape}; use 3–5 unique axes "
            "from variable, section, time, height, width, including time/height/width"
        )
    for axis in target:
        if axis not in axes:
            data = np.expand_dims(data, -1)
            axes.append(axis)
    return np.ascontiguousarray(data.transpose([axes.index(axis) for axis in target]))


def read_adios(path, config):
    """Read configured variables, optionally concatenate ADIOS steps as time.

    axes describes each variable within one ADIOS step. When steps_as_time
    is true, time may be omitted: one step then represents one time frame.
    """
    try:
        from adios2 import FileReader
    except ImportError as exc:
        raise ImportError("ADIOS input requires the optional 'adios2' package") from exc
    names = config.get("variables")
    if isinstance(names, str):
        names = [names]
    if not names or not config.get("axes"):
        raise ValueError("adios.variables and adios.axes must be configured")
    axes = list(config["axes"])
    steps_as_time = config.get("steps_as_time", False)
    if len(names) > 1 and "variable" in axes:
        raise ValueError("Multiple named variables cannot also have a variable axis")
    arrays = []
    with FileReader(str(path)) as reader:
        available = reader.available_variables()
        for name in names:
            if name not in available:
                raise ValueError(f"ADIOS variable {name!r} not found; available: {list(available)}")
            total = int(available[name]["AvailableStepsCount"])
            selection = config.get("step_range", [0, total] if steps_as_time else [0, 1])
            start, stop = selection
            if not 0 <= start < stop <= total:
                raise ValueError(f"Invalid step_range {selection} for {name!r} ({total} steps)")
            if not steps_as_time and stop - start != 1:
                raise ValueError("Enable steps_as_time to read multiple ADIOS steps")
            frames = []
            for step in range(start, stop):
                value = np.asarray(reader.read(name, step_selection=[step, 1]))
                step_axes = axes
                if steps_as_time and "time" not in axes:
                    value = np.expand_dims(value, 0)
                    step_axes = ["time", *axes]
                frames.append(canonicalize(value, step_axes))
            arrays.append(np.concatenate(frames, axis=2))
    try:
        return np.concatenate(arrays, axis=0)
    except ValueError as exc:
        raise ValueError("ADIOS variables must have matching section/time/height/width shapes") from exc
