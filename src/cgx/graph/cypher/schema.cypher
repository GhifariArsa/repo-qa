// Uniqueness constraints: one node per (repo, label, qualified name), keyed by
// the composite `id`. These also create backing indexes.
CREATE CONSTRAINT repository_id IF NOT EXISTS FOR (n:REPOSITORY) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT module_id IF NOT EXISTS FOR (n:MODULE) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT class_id IF NOT EXISTS FOR (n:CLASS) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT function_id IF NOT EXISTS FOR (n:FUNCTION) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT method_id IF NOT EXISTS FOR (n:METHOD) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT field_id IF NOT EXISTS FOR (n:FIELD) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT global_variable_id IF NOT EXISTS FOR (n:GLOBAL_VARIABLE) REQUIRE n.id IS UNIQUE;

// Every node also carries :CodeNode so questions can traverse without knowing
// the label up front, and so repo-scoped lookups hit one index.
CREATE INDEX codenode_repo IF NOT EXISTS FOR (n:CodeNode) ON (n.repo_id);
CREATE INDEX codenode_qname IF NOT EXISTS FOR (n:CodeNode) ON (n.qname);
CREATE INDEX codenode_name IF NOT EXISTS FOR (n:CodeNode) ON (n.name);
CREATE INDEX codenode_repo_qname IF NOT EXISTS FOR (n:CodeNode) ON (n.repo_id, n.qname);
CREATE INDEX codenode_repo_name IF NOT EXISTS FOR (n:CodeNode) ON (n.repo_id, n.name);
CREATE INDEX codenode_file IF NOT EXISTS FOR (n:CodeNode) ON (n.file_path);

// Class/field lookups by owning class are common in tier-1 and tier-3 queries.
CREATE INDEX method_class IF NOT EXISTS FOR (n:METHOD) ON (n.class_qname);
CREATE INDEX field_class IF NOT EXISTS FOR (n:FIELD) ON (n.class_qname);
