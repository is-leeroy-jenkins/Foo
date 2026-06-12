# User Guide

Foo organizes data acquisition and preparation around a small set of module families.

## Core workflow

1. Load files or remote content into document objects.
2. Fetch or scrape external web, scientific, civic, geospatial, environmental, or astronomical sources.
3. Normalize, inspect, and manage local data with SQLite-backed helpers.
4. Use generator wrappers to submit prompts to supported model providers.
5. Persist results or generated text through writer utilities.

## Documentation standard

The API reference is generated from Google-style docstrings. Public modules, classes, functions, methods, and properties should include a meaningful `Purpose:` section, complete `Args:` entries for public parameters, and a `Returns:` section only when the callable returns a meaningful object.
