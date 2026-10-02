"""Run the whole cleaning pipeline: python -m src.clean  (flatten -> clean)."""
from src.clean import clean, flatten

flatten.main()
clean.main()
