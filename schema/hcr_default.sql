-- HCR Default Schema
-- Schema for Highly Cited Researchers dataset tables, based on Incites export schema
-- This defines the standard structure for HCR comparison tables

CREATE TABLE IF NOT EXISTS "{table_name}" (
    name VARCHAR NOT NULL,
    percent_docs_cited DOUBLE,
    web_of_science_documents INTEGER,
    rank INTEGER,
    times_cited INTEGER,
    affiliation VARCHAR,
    web_of_science_researcherid VARCHAR,
    category_normalised_citation_impact DOUBLE,
    orcid VARCHAR,
    highly_cited_papers INTEGER,
    hot_papers INTEGER,
    esi_field VARCHAR NOT NULL,
    indicative_cross_field_score DOUBLE
);