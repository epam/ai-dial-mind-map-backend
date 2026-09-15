import base64
import json
import os
from typing import Any, Dict

from aidial_sdk.chat_completion import ChatCompletion, Request, Response
from aidial_sdk.exceptions import HTTPException, InvalidRequestError
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from openai import RateLimitError
from pydantic.types import SecretStr

from dial_rag.app import QA_CHAIN_CONFIG
from dial_rag.dial_config import DialConfig
from dial_rag.dial_user_limits import get_user_limits_for_model
from dial_rag.document_record import DocumentRecord
from dial_rag.index_storage import SERIALIZATION_CONFIG
from dial_rag.request_context import RequestContext
from dial_rag.resources.dial_limited_resources import DialLimitedResources
from mindmap.completion.rag import Rag
from mindmap.config import DIAL_URL
from mindmap.dial.batch_file_reader import BatchFileReader
from mindmap.dial.client import DialClient
from mindmap.models.graph import Graph, GraphData
from mindmap.utils import graph_patch
from mindmap.utils.dial_api import build_references, run_chain
from mindmap.utils.docstore import decode_docstore
from mindmap.utils.errors import prettify_rate_limit
from mindmap.utils.file_cache import FileCache
from mindmap.utils.graph import get_subgraph

# Bounded by memory, not by the number of entries: a single index file of a
# pdf mindmap can be tens of megabytes, so 2000 entries were unbounded in
# practice and the memory was never released.
file_cache = FileCache()


def doc_to_attach(document: Document, index=None) -> dict:
    if index is None:
        index = document.metadata.get("chunk_id", 0)

    return dict(
        type="text/markdown",
        title="[{index}] '{title}'".format(**document.metadata, index=index),
        data=f"{document.page_content}",
        reference_url=document.metadata["source"],
    )


async def get_subgraph_logic(
    client: DialClient, request: Dict[str, Any]
) -> Dict[str, Any]:
    depth = request["depth"]
    node = request.get("node", None)
    previous_node = request.get("previous_node", None)
    max_nodes = request.get("max_nodes", 19)

    await client.read_metadata()

    file_reader = BatchFileReader(client, file_cache)

    if (
        not len(client._metadata.model_fields_set)
        or not client._metadata.edges_file
        or not client._metadata.nodes_file
        or not client._metadata.documents_file
    ):
        raise HTTPException(status_code=404, message="Not found mindmap")

    file_reader.add_file(client._metadata.nodes_file)
    file_reader.add_file(client._metadata.edges_file)
    file_reader.add_file(client._metadata.documents_file)

    nodes = [
        GraphData.model_validate(node) for node in request.get("nodes", [])
    ]
    edges = [
        GraphData.model_validate(edge) for edge in request.get("edges", [])
    ]

    root = None
    docs = []
    for result in await file_reader.read():
        if result[0] == client._metadata.nodes_file:
            root = result[1]["root"]
            nodes = nodes + [
                GraphData.model_validate(node) for node in result[1]["nodes"]
            ]
        elif result[0] == client._metadata.edges_file:
            edges = edges + [
                GraphData.model_validate(edge) for edge in result[1]["edges"]
            ]
        else:
            docs = result[1]["documents"]

    file_reader = BatchFileReader(client, file_cache)

    storage_url_to_id = {}
    for i, source in enumerate(docs):
        if "storage_url" in source:
            storage_url_to_id[source["storage_url"]] = i
            file_reader.add_file(source["storage_url"])

    for result in await file_reader.read():
        docs[storage_url_to_id[result[0]]] = result[1]

    graph_data = Graph(nodes + edges)

    if not node:
        if not root:
            raise HTTPException(status_code=404, message="Not found node")

        node = root

    original_nodes = nodes

    nodes, edges = get_subgraph(
        graph_data,
        node,
        depth,
        max_nodes,
        previous_node,
        str(client._metadata.params.get("type", "universal")),
    )

    clear_nodes = [node.model_dump(exclude_none=True) for node in nodes]

    for node_obj in clear_nodes:
        if node_obj["data"]["id"] == node:
            node_obj["data"]["references"] = await build_references(
                node_obj["data"]["details"], docs, original_nodes, client
            )

    return {
        "nodes": [node for node in clear_nodes],
        "edges": [edge.model_dump(exclude_none=True) for edge in edges],
    }


class Mindmap(ChatCompletion):
    def __init__(self, dial_url: str) -> None:
        super().__init__()

        self.dial_url = dial_url

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        properties = await request.request_dial_application_properties()
        assert properties is not None

        client = await DialClient.create_without_request(
            DIAL_URL or "", request.headers.get("etag", ""), properties
        )

        if (
            request.custom_fields
            and request.custom_fields.configuration
            and "subgraph_request" in request.custom_fields.configuration
        ):
            with response.create_single_choice() as choice:
                choice.add_attachment(
                    type="application/json",
                    title="subgraph",
                    data=json.dumps(
                        await get_subgraph_logic(
                            client,
                            request.custom_fields.configuration[
                                "subgraph_request"
                            ],
                        )
                    ),
                )

            return

        force_answer_generation = (
            request.custom_fields is not None
            and request.custom_fields.configuration is not None
            and request.custom_fields.configuration.get(
                "force_answer_generation", False
            )
        )

        target_node_id = (
            request.custom_fields is not None
            and request.custom_fields.configuration is not None
            and request.custom_fields.configuration.get("target_node_id", None)
        )

        if not isinstance(force_answer_generation, bool):
            raise InvalidRequestError(
                '"force_answer_generation" parameter should be boolean'
            )

        await client.read_metadata()

        file_reader = BatchFileReader(client, file_cache)

        # TODO: check the case
        assert client._metadata.nodes_file
        assert client._metadata.edges_file
        assert client._metadata.documents_file

        file_reader.add_file(client._metadata.nodes_file)
        file_reader.add_file(client._metadata.edges_file)
        file_reader.add_file(client._metadata.documents_file)

        for node_id, file_name in client._metadata.nodes.items():
            if node_id == target_node_id:
                continue

            file_reader.add_file(file_name)

        for _, file_name in client._metadata.documents.items():
            file_reader.add_file(file_name)

        doc_file_to_doc_id = {
            file: id for id, file in client._metadata.documents.items()
        }

        nodes, edges, docs = [], [], []
        chunks_by_doc = {}
        records_by_doc = {}
        nodes_docstore: FAISS | None = None
        patch_docstore: FAISS | None = None
        records = []
        for result in await file_reader.read():
            if result[0] == client._metadata.nodes_file:
                nodes = [
                    GraphData.model_validate(node)
                    for node in result[1]["nodes"]
                ]
            elif result[0] == client._metadata.edges_file:
                edges = [
                    GraphData.model_validate(edge)
                    for edge in result[1]["edges"]
                ]
            elif result[0] == client._metadata.documents_file:
                docs = result[1]["documents"]
            elif result[0].startswith("nodes/"):
                node = result[1]

                docstore = decode_docstore(node["docstore"])

                if nodes_docstore:
                    nodes_docstore.merge_from(docstore)
                else:
                    nodes_docstore = docstore

                patch_docstore_by_node = graph_patch.decode_docstore(
                    node["patcher_docstore"]
                )

                if patch_docstore:
                    patch_docstore.merge_from(patch_docstore_by_node)
                else:
                    patch_docstore = patch_docstore_by_node
            else:
                doc = result[1]

                chunks_by_doc[doc_file_to_doc_id[result[0]]] = doc["chunks"]

                if "rag_record" in doc:
                    records_by_doc[doc_file_to_doc_id[result[0]]] = (
                        DocumentRecord.from_bytes(
                            base64.b64decode(doc["rag_record"]),
                            **SERIALIZATION_CONFIG,
                        )
                    )

        file_reader = BatchFileReader(client, file_cache)

        storage_url_to_id = {}
        for i, source in enumerate(docs):
            if "storage_url" in source:
                storage_url_to_id[source["storage_url"]] = i
                file_reader.add_file(source["storage_url"])

        for result in await file_reader.read():
            docs[storage_url_to_id[result[0]]] = result[1]

        file_reader = BatchFileReader(client, file_cache)
        index_file_to_doc_id = {}
        for doc in docs:
            if not doc.get("active", True) or doc["status"] != "INDEXED":
                continue

            if "index" in doc:
                if doc["id"] in chunks_by_doc:
                    del chunks_by_doc[doc["id"]]
                file_reader.add_file(doc["index"])
                index_file_to_doc_id[doc["index"]] = doc["id"]
            else:
                if doc["id"] in records_by_doc:
                    records.append(records_by_doc[doc["id"]])

        for result in await file_reader.read():
            doc_id = index_file_to_doc_id[result[0]]
            doc = result[1]

            chunks_by_doc[doc_id] = doc["chunks"]

            if "rag_record" in doc:
                records.append(
                    DocumentRecord.from_bytes(
                        base64.b64decode(doc["rag_record"]),
                        **SERIALIZATION_CONFIG,
                    )
                )

        graph_data = Graph(nodes + edges)

        with response.create_single_choice() as choice:
            qa_chain_config = QA_CHAIN_CONFIG
            rag_model = str(
                client._metadata.params.get("chat_model")
                or os.getenv("RAG_MODEL", default="gpt-4.1-2025-04-14")
            )
            qa_chain_config.chat_chain_config.llm_config.model_deployment_name = (
                rag_model
            )
            qa_chain_config.query_chain_config.llm_config.model_deployment_name = (
                rag_model
            )
            self.rag = Rag(
                nodes_docstore,
                patch_docstore,
                rag_model,
                graph_data,
                records,
                RequestContext(
                    dial_url=self.dial_url,
                    api_key=request.api_key,
                    chat_guardrails_enabled=bool(
                        client._metadata.params.get(
                            "chat_guardrails_enabled", False
                        )
                    ),
                    chat_prompt=str(
                        client._metadata.params.get("chat_prompt", "")
                    ),
                    chat_guardrails_prompt=str(
                        client._metadata.params.get(
                            "chat_guardrails_prompt", ""
                        )
                    ),
                    chat_guardrails_response_prompt=str(
                        client._metadata.params.get(
                            "chat_guardrails_response_prompt", ""
                        )
                    ),
                    choice=choice,
                    dial_limited_resources=DialLimitedResources(
                        lambda model_name: get_user_limits_for_model(
                            DialConfig(
                                dial_url=self.dial_url,
                                api_key=SecretStr(request.api_key),
                            ),
                            model_name,
                        )
                    ),
                ),
                qa_chain_config=qa_chain_config,
            )

            rag_chain = self.rag.create_chain(
                records,
                self.dial_url,
                SecretStr(request.api_key),
                force_answer_generation,
            )

            try:
                await run_chain(
                    choice,
                    rag_chain,
                    {"messages": request.messages},
                    content_key="answer",
                    attachment_keys=["attachment", "attachment_graph"],
                    docs=docs,
                    nodes=nodes,
                    client=client,
                    records=records,
                    chunks_by_doc=chunks_by_doc,
                )
            except RateLimitError as e:
                raise HTTPException(
                    status_code=429,
                    type="tokens limit",
                    message=prettify_rate_limit(e),
                )
