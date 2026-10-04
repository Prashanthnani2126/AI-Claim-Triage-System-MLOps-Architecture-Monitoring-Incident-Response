class Config:
    """Basic configuration for the starter prototype.

    This file is intentionally simple. Students may improve configuration
    management as part of the assessment.
    """

    # Input text columns
    TICKET_SUMMARY = "Ticket Summary"
    INTERACTION_CONTENT = "Interaction content"

    # Label columns after renaming original Type 1-Type 4 columns
    TYPE_COLS = ["y2", "y3", "y4"]

    # Core assessment target label.
    # y3 and y4 are available in the dataset but are not required for the core task.
    CLASS_COL = "y2"

    # Used by the existing prototype to run separate experiments per Type 1 group.
    GROUPED = "y1"
