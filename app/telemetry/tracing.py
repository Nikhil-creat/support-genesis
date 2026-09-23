"""OpenTelemetry tracing setup plus LangSmith environment wiring.

LangSmith is enabled purely through environment variables (its LangChain
integration auto-instruments graph/LLM calls), while OpenTelemetry gives us
vendor-neutral spans for the FastAPI layer and any custom instrumentation
(node latency, tool call spans) that we want visible in Jaeger/Tempo/etc.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import get_settings

_initialized = False


def configure_telemetry() -> None:
    global _initialized
    if _initialized:
        return
    settings = get_settings()

    resource = Resource(attributes={SERVICE_NAME: settings.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)

    _initialized = True


def get_tracer():
    return trace.get_tracer("support-engine")


@contextmanager
def traced_node(node_name: str, **attributes):
    tracer = get_tracer()
    with tracer.start_as_current_span(node_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
        yield span
