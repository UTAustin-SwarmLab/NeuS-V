def parse_tl_formula(tl_spec: str) -> str:
    """Validate the tl specification."""
    if 'G "' in tl_spec:
        tl_spec = tl_spec.replace('G "', 'F "')
    tl_spec = tl_spec.replace("-", "_")
    if tl_spec[0] == "\n":
        tl_spec = tl_spec[1:]

    if tl_spec[0] in ["F"]:
        return f"P=? [{tl_spec}]"

    if tl_spec[0] in ["G"]:
        tl_spec = tl_spec[1:]
        return f"P=? [F {tl_spec}]"

    # if any(op in tl_spec for op in ["F", "G", "U"]):
    #     return f"P=? [F ({tl_spec})]"

    return f"P=? [F {tl_spec}]"


def parse_proposition_set(proposition_set: list[str]) -> list[str]:
    """Parse the proposition set."""
    return [prop.replace("-", "_") for prop in proposition_set]


def parse_tl_specification(tl_spec: str) -> str:
    """Parse the tl specification."""
    return tl_spec.replace("-", "_")
