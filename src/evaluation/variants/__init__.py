"""Retrieval variants for the evaluation harness.

Each variant is a thin wrapper around the existing retrieval stack with the
``exclude_citing_paper_id`` plumbed through to the SQL ``WHERE`` clause.
"""
