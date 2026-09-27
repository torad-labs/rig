# test/

The in-memory machine. `fakes/index.ts` implements every port in `src/shared/ports/` as a
small class you can read (`InMemoryFileSystem`, `FakeShell`, `FakeGpu`, `FakeSystemd`, `FakeRental`,
`FakeSsh`, ...), and `fakePorts()` hands a use case the whole set; `fakes/gguf.ts` writes tiny
GGUF fixtures for the derive tests. No mocking library.

**Belongs here:** a fake for a port, and fixture builders shared by more than one slice's
tests.

**Does not belong here:** a test. Tests live beside the code they test (`*.test.ts` in `src/`;
`bunfig.toml` sets the test root to `src/`, so a test placed here would not run). Not here
either: a fixture only one slice uses (it goes in that slice), a real adapter, anything that
touches a card, a model, a network or the filesystem.

The lint allows exactly one import across the `src/` boundary: a test reaching `test/fakes/`.
