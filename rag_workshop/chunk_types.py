"""
Shared data type for corpus chunks, used by both notebook 1 (which builds and pickles
`chunks.pkl`) and the Streamlit apps (which unpickle it). It has to live in a real, importable
module rather than inside a notebook cell: pickle stores classes by their module path, and a
class defined in a notebook cell is recorded under `__main__` -- which points to the *kernel's*
main module, not to whatever script later tries to unpickle the file.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str            # raw chunk text -- exact wording, shown to the user / the LLM
    embed_text: str       # what we actually embed: a contextual header + text (see 3.4)
    heading_path: str      # breadcrumb, e.g. "SLURM Job Submission > Job Arrays"
    source_url: str
    source_title: str
    chunk_index: int  # position within the source document
