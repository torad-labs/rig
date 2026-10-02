# @rig/testing

The in-memory machine. `src/fakes.ts` implements every port in `@rig/core` as a small class you
can read (`InMemoryFileSystem`, `FakeShell`, `FakeGpu`, `FakeSystemd`, `FakeRental`, `FakeSsh`,
...), and `fakePorts()` hands a use case the whole set; `src/gguf.ts` writes tiny GGUF fixtures for
the derive tests; `src/registry.ts` answers pulls with the registry Worker itself; `repoRoot` is
the checkout, for a test that reads a committed head or the engine pin. No mocking library.

**Belongs here:** a fake for a port, and fixture builders shared by more than one package's tests.

**Does not belong here:** a test (tests live beside the code they test, `*.test.ts` in each
package's `src/`); a fixture only one package uses (it goes in that package); a real adapter
(`@rig/adapters`); anything that touches a card, a model, a network or the filesystem.

Every package lists `@rig/testing` as a devDependency, and `tools/lint-architecture.ts` refuses
code that ships importing it.
