# Everywhere - Local, painless semantic search

Search is good. Search could be better. Let's make it better.

## Why this exists

Document understanding has come a long way, thanks to modern AI. With it, we got a swarm of toolkits, frameworks and APIs that want to do them all, but few have dared to do as little as possible.

The project was inspired by [Everything](https://www.voidtools.com/support/everything/), a filename search engine for Windows that is outstanding in its ability to index and let you search your entire filesystem in real time, all while being trivial to set up and use. **Everywhere**'s goal is enable system-wide semantic search while staying true to those same principles.

## What it does

**Everywhere** currently has two main components:

1. A Python core library that handles file parsing, indexing and retrieval

2. A CLI app for interacting with the library as an end user.
