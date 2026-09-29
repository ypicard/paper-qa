import asyncio
import pickle

import pytest
from lmi import EmbeddingModel
from pydantic import Field, PrivateAttr

from paperqa import Doc, Docs, Settings, Text
from paperqa.types import PQASession


class LocalEmbedding(EmbeddingModel):
    name: str = "local"
    fail: bool = False
    block: bool = False
    requests: list[list[str]] = Field(default_factory=list)
    _started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(texts)
        self._started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("embedding failed")
        return [[1.0, float(len(text))] for text in texts]

    async def wait_until_started(self) -> None:
        await self._started.wait()


@pytest.mark.asyncio
async def test_corpus_operations_preserve_inputs_and_reuse_state():
    doc = Doc(docname="paper", dockey="one", citation="Citation")
    text = Text(name="paper 1", text="Some evidence", doc=doc)
    empty = Docs()
    settings = Settings(
        parsing={"defer_embedding": True}, answer={"evidence_skip_summary": True}
    )
    before = pickle.dumps((empty, doc, text))
    corpus, added = await empty.aadd_texts([text], doc, settings)
    assert added
    assert pickle.dumps((empty, doc, text)) == before
    before = pickle.dumps(corpus)
    session = PQASession(question="evidence")
    session_before = pickle.dumps(session)
    ready, evidence = await corpus.aget_evidence(
        session, settings=settings, embedding_model=LocalEmbedding()
    )
    assert evidence.contexts
    assert len(ready.texts_index) == 1
    assert ready.texts[0].embedding is not None
    assert pickle.dumps(corpus) == before
    assert pickle.dumps(session) == session_before
    ready_before = pickle.dumps(ready)
    model = LocalEmbedding()
    again, matches = await ready.retrieve_texts("evidence", 1, embedding_model=model)
    assert model.requests == [["evidence"]]
    assert matches
    assert again.texts[0] is ready.texts[0]
    assert pickle.dumps(ready) == ready_before

    duplicate_name = doc.model_copy(update={"dockey": "two"})
    duplicate_text = text.model_copy(update={"doc": duplicate_name})
    inputs_before = pickle.dumps((duplicate_name, duplicate_text))
    left, right = await asyncio.gather(
        ready.aadd_texts([duplicate_text], duplicate_name, settings),
        ready.aadd_texts([duplicate_text], duplicate_name, settings),
    )
    assert left[0].docs["two"].docname == "papera"
    assert right[0].docs["two"].docname == "papera"
    assert pickle.dumps((duplicate_name, duplicate_text)) == inputs_before
    assert pickle.dumps(ready) == ready_before
    assert left[0].docs is not right[0].docs
    assert left[0].texts[-1] is not right[0].texts[-1]
    assert not ready.delete(dockey="one").texts
    assert not ready.clear_docs().texts
    assert pickle.dumps(ready) == ready_before


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["insert", "retrieve"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_failure_and_cancellation_leave_inputs_unchanged(operation, cancel):
    doc = Doc(docname="paper", dockey="one", citation="Citation")
    text = Text(name="paper 1", text="Some evidence", doc=doc)
    corpus = Docs(docs={doc.dockey: doc}, docnames={doc.docname}, texts=[text])
    new_doc = doc.model_copy(update={"dockey": "two"})
    new_text = text.model_copy(update={"doc": new_doc})
    before = pickle.dumps((corpus, new_doc, new_text))
    model = LocalEmbedding(fail=not cancel, block=cancel)
    call = (
        corpus.aadd_texts([new_text], new_doc, embedding_model=model)
        if operation == "insert"
        else corpus.retrieve_texts("question", 1, embedding_model=model)
    )
    task = asyncio.create_task(call)
    await model.wait_until_started()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert pickle.dumps((corpus, new_doc, new_text)) == before


@pytest.mark.asyncio
async def test_partitioned_retrieval_isolates_concurrent_calls():
    doc = Doc(docname="paper", dockey="one", citation="Citation")
    corpus = Docs(
        docs={doc.dockey: doc},
        docnames={doc.docname},
        texts=[
            Text(name=f"paper {i}", text=str(i), doc=doc, embedding=[1.0, i + 1.0])
            for i in range(4)
        ],
    )
    ready, _ = await corpus.retrieve_texts("q", 4, embedding_model=LocalEmbedding())
    before = pickle.dumps(ready)
    results = await asyncio.gather(
        *(
            ready.retrieve_texts(
                "q",
                4,
                embedding_model=LocalEmbedding(),
                partitioning_fn=lambda text, parity=parity: (int(text.text) + parity)
                % 2,
            )
            for parity in (0, 1)
        )
    )
    assert all(len(matches) == 4 for _, matches in results)
    assert results[0][0].texts_index is not results[1][0].texts_index
    assert pickle.dumps(ready) == before


@pytest.mark.asyncio
async def test_mixed_embeddings_and_reinsert_after_deletion():
    doc = Doc(docname="paper", dockey="one", citation="Citation")
    texts = [
        Text(name="paper 1", text="cached", doc=doc, embedding=[1.0, 2.0]),
        Text(name="paper 2", text="missing", doc=doc),
    ]
    model = LocalEmbedding()
    corpus, added = await Docs().aadd_texts(texts, doc, embedding_model=model)
    assert added
    assert model.requests == [["missing"]]
    assert texts[1].embedding is None
    ready, _ = await corpus.retrieve_texts("q", 2, embedding_model=model)
    replacement = texts[0].model_copy(update={"text": "replacement"})
    replaced, _ = await ready.delete(dockey="one").aadd_texts([replacement], doc)
    _, matches = await replaced.retrieve_texts("q", 1, embedding_model=model)
    assert [match.text for match in matches] == ["replacement"]
    assert [text.text for text in ready.texts] == ["cached", "missing"]


@pytest.mark.asyncio
async def test_qdrant_retrieval_preserves_remote_collection():
    from paperqa.llms import QdrantVectorStore

    doc = Doc(docname="paper", dockey="one", citation="Citation")
    text = Text(name="paper 1", text="evidence", doc=doc, embedding=[1.0, 2.0])
    store = QdrantVectorStore()
    try:
        await store.add_texts_and_embeddings([text])
        corpus = Docs(texts_index=store)
        ready, matches = await corpus.retrieve_texts(
            "q", 1, embedding_model=LocalEmbedding()
        )
        assert matches[0].text == "evidence"
        assert corpus.texts_index is store
        assert ready.texts_index is not store
        assert (await store.client.count(store.collection_name)).count == 1
        assert len(store) == 1
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_summary_failure_preserves_corpus_and_session(monkeypatch, cancel):
    doc = Doc(docname="paper", dockey="one", citation="Citation")
    corpus, _ = await Docs().aadd_texts(
        [Text(name="paper 1", text="Some evidence", doc=doc)],
        doc,
        settings=Settings(parsing={"defer_embedding": True}),
    )
    session = PQASession(question="evidence", token_counts={"local": [1, 2]})
    before = pickle.dumps((corpus, session))
    started = asyncio.Event()

    async def fail_summary(**kwargs):
        assert kwargs["text"].embedding is not None
        started.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError("summary failed")

    monkeypatch.setattr("paperqa.docs.map_fxn_summary", fail_summary)
    task = asyncio.create_task(
        corpus.aget_evidence(session, embedding_model=LocalEmbedding())
    )
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert pickle.dumps((corpus, session)) == before


@pytest.mark.asyncio
async def test_merge_preserves_independent_additions():
    source = Docs()
    branches = []
    settings = Settings(parsing={"defer_embedding": True})
    for key in ("one", "two"):
        doc = Doc(docname="paper", dockey=key, citation="Citation")
        branch, _ = await source.aadd_texts(
            [Text(name="paper 1", text=key, doc=doc)], doc, settings
        )
        branches.append(branch)
    before = pickle.dumps((source, branches))
    merged = source.merge(branches)
    assert set(merged.docs) == {"one", "two"}
    assert merged.docnames == {"paper", "papera"}
    assert {text.doc.docname for text in merged.texts} == merged.docnames
    assert pickle.dumps((source, branches)) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_query_preserves_inputs(monkeypatch, outcome):
    from lmi import LLMModel, LLMResult

    doc = Doc(docname="paper", dockey="one", citation="Citation")
    settings = Settings(
        parsing={"defer_embedding": True}, answer={"evidence_skip_summary": True}
    )
    corpus, _ = await Docs().aadd_texts(
        [Text(name="paper 1", text="Some evidence " * 20, doc=doc)], doc, settings
    )
    session = PQASession(question="evidence", token_counts={"local": [1, 2]})
    before = pickle.dumps((corpus, session))
    started = asyncio.Event()

    async def answer(*args, **kwargs):  # noqa: ARG001
        started.set()
        if outcome == "cancel":
            await asyncio.Event().wait()
        if outcome == "failure":
            raise RuntimeError("answer failed")
        return LLMResult(
            model="local", text="Answer", prompt_count=1, completion_count=1
        )

    monkeypatch.setattr(LLMModel, "call_single", answer)
    task = asyncio.create_task(
        corpus.aquery(session, settings=settings, embedding_model=LocalEmbedding())
    )
    await started.wait()
    if outcome == "success":
        updated, result = await task
        assert len(updated.texts_index) == 1
        assert result.answer == "Answer"
        assert result.token_counts["local"] == [2, 3]
    else:
        if outcome == "cancel":
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancel" else RuntimeError
        ):
            await task
    assert pickle.dumps((corpus, session)) == before
