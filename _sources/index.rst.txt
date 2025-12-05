AutoSchemaKG: Atlas-RAG
=======================

**Atlas-RAG** is the core package of the **AutoSchemaKG** framework. It enables autonomous knowledge graph (KG) construction by combining **Knowledge Graph Triple Extraction** with dynamic **Schema Induction**.

Unlike traditional pipelines that require predefined schemas, Atlas-RAG generates schemas automatically via conceptualization, allowing for the construction of high-quality KGs from unstructured text (PDFs, Markdown, etc.) and enabling zero-shot inferencing across domains.

.. note::
   This package implements the research presented in the paper: `AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora <https://arxiv.org/abs/2505.23628>`_.

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   intro/installation
   intro/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   guide/llm_providers
   guide/file_formats
   guide/configurations
   guide/vectorstores
   guide/retrieval_augmentation

.. toctree::
   :maxdepth: 2
   :caption: Advanced Usage

   example/advance_features
   example/parralel_generation

.. toctree::
   :maxdepth: 2
   :caption: Billion-Scale KG Guide

   billion_kg/existing_billion_kg


