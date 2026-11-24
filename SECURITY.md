# Security

The server only reads public data from the INSPIRE REST API and stores nothing but an embedding
cache. Points to keep in mind when deploying it:

* The Streamable HTTP endpoint has no authentication: put it behind your own gateway or ingress if it
  must not be public, and keep the default `127.0.0.1` binding for local use.
* Titles and abstracts come from third parties. The server instructions tell the model to treat them as
  data, but the client application remains responsible for prompt-injection defences.
* Report vulnerabilities privately through the repository's security advisories, not public issues.
