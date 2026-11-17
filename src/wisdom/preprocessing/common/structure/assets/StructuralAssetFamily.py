"""StructuralAssetFamily vocabulary; scientific values are unchanged."""

from enum import Enum


class StructuralAssetFamily(str, Enum):
    """Name each optional immutable structural/representation contract."""

    ATOMIC_CONTEXT = "atomic_context"
    EXTENDED_SURFACE_ATOM = "extended_surface_atom"
    SEQUENCE_CONTEXT = "sequence_context"
    COVALENT_ENVIRONMENT = "covalent_environment"
    ATOMIC_SASA = "atomic_sasa"
    PHYSICAL_REFERENCE = "physical_reference"
    PHYSICAL_QUERIES = "physical_queries"
    RESIDUE_GRAPH = "residue_graph"
    SURFACE_ENVIRONMENT = "surface_environment"
    EXTERNAL_ATOMIC_SLAE = "external_atomic_slae"
    SURFACE_LOCAL_GRAPH = "surface_local_graph"
    SURFACE_VIEW = "surface_view"
    SPECTRAL_EXTENSION = "spectral_extension"
    EXTERNAL_EMBEDDINGS = "external_embeddings"
    STRUCTURE_TOKENS = "structure_tokens"
    STATIC_EMBEDDING_TABLE = "static_embedding_table"
    EXTERNAL_EMBEDDINGS_ESM2 = "external_embeddings_esm2"
    EXTERNAL_EMBEDDINGS_ESMC = "external_embeddings_esmc"
    EXTERNAL_EMBEDDINGS_SAPROT = "external_embeddings_saprot"
    STATIC_EMBEDDING_TABLE_ESM2 = "static_embedding_table_esm2"
    STATIC_EMBEDDING_TABLE_ESMC = "static_embedding_table_esmc"
    STATIC_EMBEDDING_TABLE_SAPROT = "static_embedding_table_saprot"
