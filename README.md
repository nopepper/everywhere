# Everywhere - Fast & local semantic search

![Demo GIF](./docs/demo.gif)

## Why this exists

AI has gotten pretty good at understanding text. That's great. Agents and RAG pipelines are bulky, slow and complex. That's not so great. **Everywhere** is a simple terminal app that uses fast, on-device methods to index and let you search your entire filesystem in real time.

The project takes inspiration from [Everything](https://www.voidtools.com/support/everything/), the ultra-efficient filename search engine for Windows. While not quite as _blazingly fast_, Everywhere should be lightweight, effective and self-explanatory.

## Disclaimer

**Everywhere** is in the very early stages of development. It is not yet suitable for serious use. Expect bugs, performance issues, and breaking changes.

## Getting Started

Make sure [uv](https://github.com/astral-sh/uv) is installed on your system.

Try out the app without installing:

```bash
uvx --from git+https://github.com/nopepper/everywhere.git everywhere
```

Installing and running:

```bash
uv tool install git+https://github.com/nopepper/everywhere.git
uvx everywhere
```

Alternatively, you can clone the repository and run it from source:

```bash
git clone https://github.com/nopepper/everywhere.git
cd everywhere
uv run everywhere
```

## Usage Notes

- When you first run the app, it will start indexing the `data_test` directory in your current working directory. You can change the directories to be indexed from the command palette (click the circle in the top left corner, then select "Pick indexed directories...").
- Everything continuously monitors the filesystem for changes and updates the index accordingly.
- Index data is stored in the `cache` directory of the project.

## Roadmap

- [ ] Add more search providers
  - [ ] Full-text indexing
  - [x] PDF search (text only)
  - [x] PDF search (OCR)
  - [x] Word document search (text only)
- [x] Improve performance
  - [x] Maximize throughput on CPU
  - [x] Vector database support
  - [x] Vector database saving/loading
  - [x] File watching with [watchdog](https://github.com/gorakhargosh/watchdog)
- [ ] Improve UX
  - [x] Indexing in background thread
  - [x] Indexing progress bar
  - [x] Limit search results to 1000 items
  - [x] Aggregate search results from multiple search providers
  - [ ] Soft operators
    - [ ] Add OR operator !or (additive)
    - [ ] Add AND operator !and (multiplicative)
    - [ ] Add NOT operator !not (inverse)
  - [x] FS watcher checkbox selector
    - [x] Let user choose which directories to watch
  - [ ] Search provider settings
    - [ ] For each search provider, show checkboxes for each document type it supports
- [ ] Improve quality
  - [x] Normalize embeddings
  - [x] Add small overlap when chunking
  - [x] Chunks at 256 tokens, unless the document is very small
  - [ ] Nested RAG (successive halving)
- [ ] Improve reliability
  - [ ] Add tests
  - [x] Add logging & events
- [ ] Documentation & OSS
  - [x] License
  - [x] Readme
    - [x] Installation
    - [x] Usage
    - [x] Roadmap
    - [x] License
  - [ ] Update docs
- [ ] Future
  - [ ] GPU support
  - [ ] Faster filesystem parsing
  - [ ] Remote search providers (GDrive, Notion)
  - [ ] Image search providers

## Contributing

Feel free to open an issue if you find a bug or have an idea for a new feature.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
