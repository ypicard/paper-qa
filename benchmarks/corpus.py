"""Compare corpus operations with the former caller-side deep copy.

Run with --legacy against the parent revision and without it against this branch.
"""

import argparse
import asyncio
import json
import statistics
import time
import tracemalloc

from lmi import EmbeddingModel

from paperqa import Doc, Docs, Settings, Text


class LocalEmbedding(EmbeddingModel):
    name: str = "local"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * 256 for _ in texts]


async def main(legacy: bool) -> None:
    doc = Doc(docname="seed", dockey="seed", citation="Seed")
    corpus = Docs(
        docs={doc.dockey: doc},
        docnames={doc.docname},
        texts=[
            Text(
                name=f"seed {i}",
                text="Evidence. " * 100,
                doc=doc,
                embedding=[1.0] * 256,
            )
            for i in range(2000)
        ],
    )
    cold = corpus.model_copy(deep=True)
    model = LocalEmbedding()
    settings = Settings(
        parsing={"defer_embedding": True}, answer={"evidence_skip_summary": True}
    )
    result = await corpus.aget_evidence(
        "warm", settings=settings, embedding_model=model
    )
    if not legacy:
        corpus, _ = result
    for operation in ("insert", "cold_retrieval", "warm_retrieval"):
        samples = []
        for _ in range(5):
            tracemalloc.start()
            start = time.perf_counter()
            source = cold if operation == "cold_retrieval" else corpus
            working = source.model_copy(deep=True) if legacy else source
            if operation == "insert":
                added = Doc(docname="added", dockey="added", citation="Added")
                await working.aadd_texts(
                    [
                        Text(
                            name="added 1",
                            text="New evidence",
                            doc=added,
                            embedding=[1.0] * 256,
                        )
                    ],
                    added,
                    settings=settings,
                )
            else:
                await working.aget_evidence(
                    "question", settings=settings, embedding_model=model
                )
            samples.append(
                (time.perf_counter() - start, tracemalloc.get_traced_memory()[1])
            )
            tracemalloc.stop()
        print(
            json.dumps(
                {
                    "operation": operation,
                    "median_ms": statistics.median(s[0] for s in samples) * 1000,
                    "peak_mib": statistics.median(s[1] for s in samples) / 1024**2,
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    asyncio.run(main(parser.parse_args().legacy))
