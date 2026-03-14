import pytest
import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

# We will implement this class in vector_engine.py
# from memory.vector_engine import VectorEngine 

@pytest.mark.asyncio
async def test_vector_engine_anchored_retrieval():
    """
    RED TEST: Should retrieve the raw 3-turn anchor when a semantic key is matched.
    """
    # This will fail because VectorEngine is not yet defined
    from memory.vector_engine import VectorEngine

    # Keep this as a deterministic unit test by mocking embeddings.
    with patch.object(VectorEngine, "_get_embedding", new=AsyncMock(return_value=[0.1] * 768)):
        ve = VectorEngine(session_id="test_session")
        await ve.clear()

        # 1. Index a "Key" anchored to a "Raw Turn"
        key = "User preference: Strictly uses dark mode for all UI components"
        anchor = "User: I only ever want to see dark mode. Assistant: Understood, I will apply dark mode."

        await ve.index(key=key, anchor=anchor, metadata={"importance": 1.0})

        # 2. Query with a different but semantically similar string
        results = await ve.query("What are the UI theme preferences?", limit=1)

        assert len(results) > 0
        assert results[0]["key"] == key
        assert results[0]["anchor"] == anchor

@pytest.mark.asyncio
async def test_vector_engine_multi_key_same_anchor():
    """
    Should preserve key→anchor relationships for multiple keys sharing one anchor.
    """
    from memory.vector_engine import VectorEngine

    with patch.object(VectorEngine, "_get_embedding", new=AsyncMock(return_value=[0.1] * 768)):
        ve = VectorEngine(session_id="test_session_unique")
        await ve.clear()

        anchor = "Static conversation turn"
        await ve.index(key="key1", anchor=anchor)
        await ve.index(key="key2", anchor=anchor)  # Same anchor, different key

        count = await ve.count()
        assert count == 2


@pytest.mark.asyncio
async def test_vector_engine_uniqueness():
    """
    Should prevent duplicate indexing of identical key+anchor pairs.
    """
    from memory.vector_engine import VectorEngine

    with patch.object(VectorEngine, "_get_embedding", new=AsyncMock(return_value=[0.1] * 768)):
        ve = VectorEngine(session_id="test_session_unique_exact")
        await ve.clear()

        anchor = "Static conversation turn"
        await ve.index(key="key1", anchor=anchor)
        await ve.index(key="key1", anchor=anchor)

        count = await ve.count()
        assert count == 1
