from __future__ import annotations

import pandas as pd


def build_star_schema_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create dimension/fact DataFrames for a star schema.

    Returns a dict of table_name -> dataframe.
    This is a placeholder to be filled when your DW design is finalized.
    """
    return {"staging": df.copy()}
