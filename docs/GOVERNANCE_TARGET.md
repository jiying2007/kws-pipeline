# Terminal governance target

The source-controlled workflows are the implementation of software gates; repository settings are the enforcement boundary.

For terminal product operation, `main` should require a pull request and the complete CI matrix, block force-push and deletion, require resolution of review conversations, and keep release tags immutable. A single-maintainer repository may omit an approving review requirement so the maintainer is not deadlocked, but direct unverified changes to released product history should not be the normal path.

Until those platform controls are enabled, documentation must describe the repository as an incubation/single-maintainer governance phase rather than a fully enforced terminal governance state.
