from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import urllib.request
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Self, cast
from uuid import UUID, uuid4

from aviary.core import Message
from lmi import Embeddable, EmbeddingModel, LLMModel
from lmi.types import set_llm_session_ids
from lmi.utils import gather_with_concurrency
from pydantic import BaseModel, ConfigDict, Field

from paperqa.clients import DEFAULT_CLIENTS, DocMetadataClient
from paperqa.core import llm_parse_json, map_fxn_summary
from paperqa.llms import (
    NumpyVectorStore,
    VectorStore,
)
from paperqa.prompts import CANNOT_ANSWER_PHRASE, EMPTY_CONTEXTS
from paperqa.readers import read_doc
from paperqa.settings import MaybeSettings, get_settings
from paperqa.types import Doc, DocDetails, DocKey, PQASession, Text
from paperqa.utils import (
    citation_to_docname,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
)

logger = logging.getLogger(__name__)


def _prepare_session(query: PQASession | str, config_md5: str) -> PQASession:
    if isinstance(query, str):
        return PQASession(question=query, config_md5=config_md5)
    return query.model_copy(
        update={
            "contexts": list(query.contexts),
            "token_counts": {
                key: list(value) for key, value in query.token_counts.items()
            },
        }
    )


class Docs(BaseModel):  # noqa: PLW1641  # TODO: add __hash__
    """A collection of documents to be used for answering questions."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    docs: dict[DocKey, Doc | DocDetails] = Field(default_factory=dict)
    texts: list[Text] = Field(default_factory=list)
    docnames: set[str] = Field(default_factory=set)
    texts_index: VectorStore = Field(default_factory=NumpyVectorStore)
    name: str = Field(default="default", description="Name of this docs collection")
    deleted_dockeys: set[DocKey] = Field(default_factory=set)

    def __eq__(self, other) -> bool:
        if (
            not isinstance(other, type(self))
            or not isinstance(self.texts_index, NumpyVectorStore)
            or not isinstance(other.texts_index, NumpyVectorStore)
        ):
            return NotImplemented
        return (
            self.docs == other.docs
            and len(self.texts) == len(other.texts)
            and all(
                self_text == other_text
                for self_text, other_text in zip(self.texts, other.texts, strict=True)
            )
            and self.docnames == other.docnames
            and self.texts_index == other.texts_index
            and self.name == other.name
            # NOTE: ignoring deleted_dockeys
        )

    def clear_docs(self) -> Self:
        """Return an empty corpus without changing this collection."""
        return self.model_copy(
            update={
                "texts": [],
                "docs": {},
                "docnames": set(),
                "texts_index": NumpyVectorStore(),
                "deleted_dockeys": set(),
            }
        )

    def _get_unique_name(self, docname: str) -> str:
        """Create a unique name given proposed name."""
        suffix = ""
        while docname + suffix in self.docnames:
            # move suffix to next letter
            suffix = "a" if not suffix else chr(ord(suffix) + 1)
        docname += suffix
        return docname

    async def aadd_file(
        self,
        file: BinaryIO,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> tuple[Self, str | None]:
        """Return the updated corpus and added document name, or None for a duplicate."""
        # just put in temp file and use existing method
        suffix = ".txt"
        if maybe_is_pdf(file):
            suffix = ".pdf"
        elif maybe_is_html(file):
            suffix = ".html"

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(file.read())
            f.seek(0)
            return await self.aadd(
                Path(f.name),
                citation=citation,
                docname=docname,
                dockey=dockey,
                title=title,
                doi=doi,
                authors=authors,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
                **kwargs,
            )

    async def aadd_url(
        self,
        url: str,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> tuple[Self, str | None]:
        """Return the updated corpus and added document name, or None for a duplicate."""
        with urllib.request.urlopen(url) as f:  # noqa: ASYNC210, S310
            # need to wrap to enable seek
            file = BytesIO(f.read())
            return await self.aadd_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )

    async def aadd(  # noqa: PLR0912
        self,
        path: str | os.PathLike,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> tuple[Self, str | None]:
        """Return the updated corpus and added document name, or None for a duplicate."""
        all_settings = get_settings(settings)
        parse_config = all_settings.parsing
        content_hash = md5sum(path)
        dockey_is_content_hash = False
        if dockey is None:
            dockey = content_hash
            dockey_is_content_hash = True
        if llm_model is None:
            llm_model = all_settings.get_llm()
        if citation is None:
            # Peek first chunk
            texts = await read_doc(
                path,
                Doc(  # Fake doc
                    docname="", citation="", dockey=dockey, content_hash=content_hash
                ),
                page_size_limit=parse_config.page_size_limit,
                parse_media=False,  # Peeking is text only
                # We only use the first chunk, so let's peek just enough pages for that.
                # Usually pages 1 - 2 give that,
                # but in the event page 2 is blank (true for some PDFs),
                # we read pages 1 - 3 to be safe
                page_range=(1, 3),
                parse_pdf=parse_config.parse_pdf,
                **parse_config.reader_config,
            )
            if not texts or not texts[0].text.strip():
                raise ValueError(f"Could not read document {path}. Is it empty?")
            result = await llm_model.call_single(
                messages=[
                    Message(
                        content=parse_config.citation_prompt.format(text=texts[0].text)
                    ),
                ],
            )
            citation = cast("str", result.text)
            if (
                len(citation) < 3  # noqa: PLR2004
                or "Unknown" in citation
                or "insufficient" in citation
            ):
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"
            del result, texts  # Ensure we don't reuse

        doc = Doc(
            docname=self._get_unique_name(
                citation_to_docname(citation) if docname is None else docname
            ),
            citation=citation,
            dockey=dockey,
            content_hash=content_hash,
        )

        # try to extract DOI / title from the citation
        if (doi is title is None) and parse_config.use_doc_details:
            # TODO: specify a JSON schema here when many LLM providers support this
            messages = [
                Message(
                    content=parse_config.structured_citation_prompt.format(
                        citation=citation
                    ),
                ),
            ]
            result = await llm_model.call_single(
                messages=messages,
            )
            # This code below tries to isolate the JSON
            # based on observed messages from LLMs
            # it does so by isolating the content between
            # the first { and last } in the response.
            # Since the anticipated structure should  not be nested,
            # we don't have to worry about nested curlies.
            clean_text = cast("str", result.text).split("{", 1)[-1].split("}", 1)[0]
            clean_text = "{" + clean_text + "}"
            try:
                citation_json = json.loads(clean_text)
                if citation_title := citation_json.get("title"):
                    title = citation_title
                if citation_doi := citation_json.get("doi"):
                    doi = citation_doi
                if citation_author := citation_json.get("authors"):
                    authors = citation_author
            except (json.JSONDecodeError, AttributeError):
                # json.JSONDecodeError: clean_text was not actually JSON
                # AttributeError: citation_json was not a dict (e.g. a list)
                logger.warning(
                    "Failed to parse all of title, DOI, and authors from the"
                    " ParsingSettings.structured_citation_prompt's response"
                    f" {clean_text}, consider using a manifest file or specifying a"
                    " different citation prompt."
                )
        # see if we can upgrade to DocDetails
        # if not, we can progress with a normal Doc
        # if "fields_to_overwrite_from_metadata" is used:
        # will map "docname" to "key", and "dockey" to "doc_id"
        if (title or doi) and parse_config.use_doc_details:
            if kwargs.get("metadata_client"):
                metadata_client = kwargs["metadata_client"]
            else:
                metadata_client = DocMetadataClient(
                    http_client=kwargs.pop("http_client", None),
                    metadata_clients=kwargs.pop("clients", DEFAULT_CLIENTS),
                )

            # Query here means a query to a metadata provider
            query_kwargs: dict[str, Any] = {}

            if doi:
                query_kwargs["doi"] = doi
            if authors:
                query_kwargs["authors"] = authors
            if title:
                query_kwargs["title"] = title
            if dockey_is_content_hash:
                # if we had an autogenerated dockey, we would like our
                # metadata to be able to overwrite it if possible
                doc.fields_to_overwrite_from_metadata = {
                    d
                    for d in doc.fields_to_overwrite_from_metadata
                    if d not in {"dockey", "doc_id"}
                }

            doc = await metadata_client.upgrade_doc_to_doc_details(
                doc, **(query_kwargs | kwargs)
            )

        parse_media, enrich_media = parse_config.should_parse_and_enrich_media
        multimodal_kwargs: dict[str, Any] = {"parse_media": parse_media}
        if enrich_media:
            multimodal_kwargs["multimodal_enricher"] = (
                all_settings.make_media_enricher()
            )
        texts, metadata = await read_doc(
            path,
            doc,
            page_size_limit=parse_config.page_size_limit,
            parse_pdf=parse_config.parse_pdf,
            include_metadata=True,
            **multimodal_kwargs,
            **parse_config.reader_config,
        )
        # loose check to see if document was loaded
        if metadata.name != "image" and (
            not texts
            or len(texts[0].text) < 10  # noqa: PLR2004
            or (
                not parse_config.disable_doc_valid_check
                and (
                    (
                        # Quick sanity check the text is not just some terse one-page
                        # 404 message interspersed with newlines. Check here
                        # instead of maybe_is_text because a 404 HTML page is text
                        sum(len(t.text.replace("\n", "")) for t in texts[:2])
                        < 20  # noqa: PLR2004
                    )
                    # Use the first few text chunks to avoid potential issues with
                    # title page parsing in the first chunk
                    or not maybe_is_text("".join(t.text for t in texts[:5]))
                )
            )
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Pass disable_check"
                " to ignore this error."
            )
        updated, added = await self.aadd_texts(
            texts, doc, all_settings, embedding_model
        )
        return updated, updated.docs[doc.dockey].docname if added else None

    async def aadd_texts(
        self,
        texts: list[Text],
        doc: Doc,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> tuple[Self, bool]:
        """
        Return a corpus containing the chunked texts without changing the inputs.

        This is useful to use if you have already chunked the texts yourself.

        Returns:
            Updated corpus and whether the document passed filters and was added.
        """
        if doc.dockey in self.docs:
            return self, False
        if not texts:
            raise ValueError("No texts to add.")

        all_settings = get_settings(settings)
        if not all_settings.parsing.defer_embedding and not embedding_model:
            # want to embed now!
            embedding_model = all_settings.get_embedding_model()

        # 0. Short-circuit if it is caught by a filter
        for doc_filter in all_settings.parsing.doc_filters or []:
            if not doc.matches_filter_criteria(doc_filter):
                return self, False

        texts = [
            text.model_copy() if text.embedding is None else text for text in texts
        ]
        to_embed = [text for text in texts if text.embedding is None]
        if embedding_model and to_embed:
            for t, t_embedding in zip(
                to_embed,
                await embedding_model.embed_documents(
                    texts=await asyncio.gather(
                        *(
                            t.get_embeddable_text(
                                all_settings.parsing.should_parse_and_enrich_media[1]
                            )
                            for t in to_embed
                        )
                    )
                ),
                strict=True,
            ):
                t.embedding = t_embedding
        return self._insert_texts(texts, doc)

    def _insert_texts(self, texts: list[Text], doc: Doc) -> tuple[Self, bool]:
        if doc.dockey in self.docs:
            return self, False
        if not texts:
            raise ValueError("No texts to add.")
        if not doc.docname or not doc.dockey:
            return self, False
        new_name = self._get_unique_name(doc.docname)
        added_doc = doc.model_copy(update={"docname": new_name})
        added_texts = [
            text.model_copy(
                update={
                    "doc": added_doc,
                    "name": text.name.replace(doc.docname, new_name),
                }
            )
            for text in texts
        ]
        return (
            self.model_copy(
                update={
                    "docs": {**self.docs, doc.dockey: added_doc},
                    "texts": [*self.texts, *added_texts],
                    "docnames": self.docnames | {new_name},
                    "deleted_dockeys": self.deleted_dockeys - {doc.dockey},
                    "texts_index": (
                        NumpyVectorStore(mmr_lambda=self.texts_index.mmr_lambda)
                        if doc.dockey in self.deleted_dockeys
                        else self.texts_index
                    ),
                }
            ),
            True,
        )

    def merge(self, sources: Sequence[Docs]) -> Self:
        """Return the union of already acquired corpora without embedding or filtering.

        Keep the first document for each key. Retrieval embeds any missing vectors.
        """
        updated = self
        for source in sources:
            texts_by_key: dict[DocKey, list[Text]] = defaultdict(list)
            for text in source.texts:
                texts_by_key[text.doc.dockey].append(text)
            for doc in source.docs.values():
                if texts := texts_by_key.get(doc.dockey):
                    updated, _ = updated._insert_texts(texts, doc)
        return updated

    def delete(
        self,
        name: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
    ) -> Self:
        """Return a corpus without the selected document."""
        # name is an alias for docname
        if name and docname and name != docname:
            raise ValueError(
                "When specifying both name and docname for deletion,"
                f" they need to match. The inputs were {name=} and {docname=}."
            )
        if name is not None:
            warnings.warn(
                "The 'name' argument is deprecated in favor of 'docname',"
                " this deprecation will conclude in version 6.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            name = docname

        if name is not None:
            doc = next((doc for doc in self.docs.values() if doc.docname == name), None)
            if doc is None:
                return self
            dockey = doc.dockey
        removed = self.docs[dockey]
        return self.model_copy(
            update={
                "docs": {key: doc for key, doc in self.docs.items() if key != dockey},
                "docnames": self.docnames - {removed.docname},
                "deleted_dockeys": self.deleted_dockeys | {dockey},
                "texts": [text for text in self.texts if text.doc.dockey != dockey],
            }
        )

    async def _build_texts_index(
        self, embedding_model: EmbeddingModel, with_enrichment: bool = False
    ) -> Self:
        texts = [t for t in self.texts if t not in self.texts_index]
        to_embed = [t for t in texts if t.embedding is None]
        replacements = {}
        if to_embed:
            embeddings = await embedding_model.embed_documents(
                texts=await asyncio.gather(
                    *(t.get_embeddable_text(with_enrichment) for t in to_embed)
                )
            )
            replacements = {
                id(text): text.model_copy(update={"embedding": embedding})
                for text, embedding in zip(to_embed, embeddings, strict=True)
            }
        index = await self.texts_index.fork()
        if texts:
            await index.add_texts_and_embeddings(
                [replacements.get(id(text), text) for text in texts]
            )
        return self.model_copy(
            update={
                "texts": [replacements.get(id(text), text) for text in self.texts],
                "texts_index": index,
            }
        )

    async def retrieve_texts(
        self,
        query: str,
        k: int,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> tuple[Self, list[Text]]:
        """Return the corpus with reusable retrieval state and matching texts."""
        settings = get_settings(settings)
        if embedding_model is None:
            embedding_model = settings.get_embedding_model()

        updated = await self._build_texts_index(
            embedding_model,
            with_enrichment=settings.parsing.should_parse_and_enrich_media[1],
        )
        updated.texts_index.mmr_lambda = settings.texts_index_mmr_lambda
        _k = k + len(self.deleted_dockeys)
        matches: list[Text] = cast(
            "list[Text]",
            (
                await updated.texts_index.max_marginal_relevance_search(
                    query,
                    k=_k,
                    fetch_k=2 * _k,
                    embedding_model=embedding_model,
                    partitioning_fn=partitioning_fn,
                )
            )[0],
        )
        matches = [m for m in matches if m.doc.dockey not in self.deleted_dockeys]
        return updated, matches[:k]

    async def aget_evidence(
        self,
        query: PQASession | str,
        settings: MaybeSettings = None,
        callbacks: Sequence[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> tuple[Self, PQASession]:

        evidence_settings = get_settings(settings)
        answer_config = evidence_settings.answer
        prompt_config = evidence_settings.prompts

        session = _prepare_session(query, evidence_settings.md5)
        updated = self

        if not self.docs and len(self.texts_index) == 0:
            return updated, session

        if embedding_model is None:
            embedding_model = evidence_settings.get_embedding_model()

        if summary_llm_model is None:
            summary_llm_model = evidence_settings.get_summary_llm()

        if answer_config.evidence_retrieval:
            updated, matches = await self.retrieve_texts(
                session.question,
                answer_config.evidence_k,
                evidence_settings,
                embedding_model,
                partitioning_fn=partitioning_fn,
            )
        else:
            matches = self.texts

        matches = (
            matches[: answer_config.evidence_k]
            if answer_config.evidence_retrieval
            else matches
        )

        prompt_templates = None
        if not answer_config.evidence_skip_summary:
            if prompt_config.use_json:
                prompt_templates = (
                    prompt_config.summary_json,
                    prompt_config.summary_json_system,
                )
            else:
                prompt_templates = (
                    prompt_config.summary,
                    prompt_config.system,
                )

        with set_llm_session_ids(session.id):
            results = await gather_with_concurrency(
                answer_config.max_concurrent_requests,
                [
                    map_fxn_summary(
                        text=m,
                        question=session.question,
                        summary_llm_model=summary_llm_model,
                        prompt_templates=prompt_templates,
                        extra_prompt_data={
                            "summary_length": answer_config.evidence_summary_length,
                            "citation": f"{m.name}: {m.doc.formatted_citation}",
                        },
                        parser=llm_parse_json if prompt_config.use_json else None,
                        callbacks=callbacks,
                        skip_citation_strip=answer_config.skip_evidence_citation_strip,
                        evidence_text_only_fallback=answer_config.evidence_text_only_fallback,
                    )
                    for m in matches
                ],
            )

        for _, llm_results in results:
            for r in llm_results:
                session.add_tokens(r)

        # Filter out failed context creations or irrelevant contexts,
        # and don't add duplicate contexts
        session.contexts += list(
            {
                c
                for c, _ in results
                if c is not None and c.score > 0 and c not in session.contexts
            }
        )
        return updated, session

    async def aquery(
        self,
        query: PQASession | str,
        settings: MaybeSettings = None,
        callbacks: Sequence[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> tuple[Self, PQASession]:
        query_settings = get_settings(settings)
        answer_config = query_settings.answer
        prompt_config = query_settings.prompts

        if llm_model is None:
            llm_model = query_settings.get_llm()
        if summary_llm_model is None:
            summary_llm_model = query_settings.get_summary_llm()
        if embedding_model is None:
            embedding_model = query_settings.get_embedding_model()

        session = _prepare_session(query, query_settings.md5)
        updated = self
        contexts = session.contexts
        if answer_config.get_evidence_if_no_contexts and not contexts:
            updated, session = await self.aget_evidence(
                session,
                callbacks=callbacks,
                settings=settings,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
                partitioning_fn=partitioning_fn,
            )
            contexts = session.contexts
        pre_str = None
        if prompt_config.pre is not None:
            with set_llm_session_ids(session.id):
                messages = [
                    Message(role="system", content=prompt_config.system),
                    Message(
                        role="user",
                        content=prompt_config.pre.format(question=session.question),
                    ),
                ]
                pre = await llm_model.call_single(
                    messages=messages,
                    callbacks=callbacks,
                    name="pre",
                )
            session.add_tokens(pre)
            pre_str = pre.text

        context_str = await query_settings.context_serializer(
            contexts=contexts,
            question=session.question,
            pre_str=pre_str,
        )

        if len(context_str.strip()) <= EMPTY_CONTEXTS:
            answer_text = (
                f"{CANNOT_ANSWER_PHRASE} this question due to"
                f" {'having no papers' if not self.docs else 'insufficient information.'}."
            )
            answer_reasoning = None
        else:
            with set_llm_session_ids(session.id):
                prior_answer_prompt = ""
                if prompt_config.answer_iteration_prompt and session.answer:
                    prior_answer_prompt = prompt_config.answer_iteration_prompt.format(
                        prior_answer=session.answer
                    )
                messages = [
                    Message(role="system", content=prompt_config.system),
                    Message(
                        role="user",
                        content=prompt_config.qa.format(
                            context=context_str,
                            answer_length=answer_config.answer_length,
                            question=session.question,
                            example_citation=prompt_config.EXAMPLE_CITATION,
                            prior_answer_prompt=prior_answer_prompt,
                        ),
                    ),
                ]
                answer_result = await llm_model.call_single(
                    messages=messages,
                    callbacks=callbacks,
                    name="answer",
                )
            answer_text = cast("str", answer_result.text)
            answer_reasoning = answer_result.reasoning_content
            session.add_tokens(answer_result)
        # it still happens
        if (ex_citation := prompt_config.EXAMPLE_CITATION) in answer_text:
            answer_text = answer_text.replace(ex_citation, "")

        if answer_config.answer_filter_extra_background:
            answer_text = re.sub(
                r"\([Ee]xtra [Bb]ackground [Ii]nformation\)",  # spellchecker: disable-line
                "",
                answer_text,
            )

        if prompt_config.post is not None:
            with set_llm_session_ids(session.id):
                messages = [
                    Message(role="system", content=prompt_config.system),
                    Message(
                        role="user",
                        content=prompt_config.post.format(question=session.question),
                    ),
                ]
                post = await llm_model.call_single(
                    messages=messages,
                    callbacks=callbacks,
                    name="post",
                )
            answer_text = cast("str", post.text)
            answer_reasoning = post.reasoning_content
            session.add_tokens(post)
            answer_text = f"{answer_text}\n\n{post.text}"

        # now at end we modify, so we could have retried earlier
        session.raw_answer = answer_text
        session.answer_reasoning = answer_reasoning
        session.contexts = contexts
        session.context = context_str

        session.populate_formatted_answers_and_bib_from_raw_answer()

        return updated, session


def merge_docs(*corpora: Docs) -> Docs:
    """Combine corpora without changing the inputs.

    The first document for each dockey wins. Later documents with colliding
    names receive unique names; look them up by dockey in the returned corpus.
    Pass the existing corpus first to retain its embeddings and retrieval index.
    No arguments returns an empty corpus. Unchanged values may be shared.
    """
    if not corpora:
        return Docs()
    return corpora[0].merge(corpora[1:])
