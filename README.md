# Everywhere - Local, painless semantic search

Search is good. Search could be better. Let's make it better.

## Why this exists

Document understanding has come a long way, thanks to modern AI. With it, we got a swarm of toolkits, frameworks and APIs that want to do them all, but few have dared to do as little as possible.

The project was inspired by [Everything](https://www.voidtools.com/support/everything/), a filename search engine for Windows that is outstanding in its ability to index and let you search your entire filesystem in real time, all while being trivial to set up and use. **Everywhere**'s goal is enable system-wide semantic search while staying true to those same principles.

## What it does

**Everywhere** currently has two main components:

1. A Python core library that handles file parsing, indexing and retrieval

2. A CLI app for interacting with the library as an end user.

## Roadmap

- [ ] Add more search providers
  - [ ] Fuzzy string search
  - [ ] PDF search (text only)
  - [ ] Word document search (text only)
- [ ] Improve performance
  - [ ] Maximize throughput on CPU
  - [ ] Vector database support & saving/loading
  - [ ] File watching with [watchdog](https://github.com/gorakhargosh/watchdog)
- [ ] Improve UX
  - [ ] Indexing in background thread
  - [ ] Indexing progress bar
  - [ ] Limit search results to 1000 items
  - [ ] Aggregate search results from multiple search providers
  - [ ] Soft operators
    - [ ] Add OR operator !or (additive)
    - [ ] Add AND operator !and (multiplicative)
    - [ ] Add NOT operator !not (inverse)
  - [ ] FS watcher checkbox selector
    - [ ] Let user choose which directories to watch
  - [ ] Search provider settings
    - [ ] For each search provider, show checkboxes for each document type it supports
- [ ] Improve quality
  - [ ] Normalize embeddings
  - [ ] Add small overlap when chunking
  - [ ] Chunks at 256 tokens, unless the document is very small
  - [ ] Nested RAG (successive halving)
- [ ] Improve reliability
  - [ ] Add tests
  - [ ] Add logging & events
- [ ] Documentation & OSS
  - [x] License
  - [ ] Readme
    - [ ] Installation
    - [ ] Usage
    - [x] Roadmap
    - [x] License
  - [ ] Update docs
- [ ] Future
  - [ ] GPU support
  - [ ] Faster filesystem parsing
  - [ ] Remote search providers (GDrive, Notion)
  - [ ] Image search providers

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
