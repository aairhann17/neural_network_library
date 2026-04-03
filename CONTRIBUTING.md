# Contributing Guide

Thanks for contributing to NeuralNetCpp.

## Development Workflow

1. Fork or branch from `main`.
2. Make focused, reviewable changes.
3. Build locally before opening a pull request.
4. Include clear commit messages and PR descriptions.

## Build and Validation

From `neural_network_library/`:

```bash
cmake -S . -B build
cmake --build build --config Release
```

For documentation generation:

```bash
cmake -S . -B build_docs -DBUILD_DOCS=ON
cmake --build build_docs --target docs --config Release
```

## Documentation Standards

Public APIs should include Doxygen comments that are precise and complete.

- Use `@brief` for every public class, function, and typedef where relevant.
- Add `@param` for all non-trivial parameters.
- Add `@return` for non-void functions.
- Add `@throws` where validation or runtime checks can fail.
- Keep comments behavior-focused and avoid repeating obvious syntax.

Group placement should remain consistent with `neural_network_library/docs/api_groups.dox`:

- `tensor_api`
- `module_api`
- `sequential_api`
- `activation_api`
- `loss_api`
- `optimizer_api`

## Code Style

- Follow existing formatting and naming in nearby files.
- Keep changes minimal and scoped to the task.
- Prefer readable, explicit error messages for invalid input paths.

## Pull Requests

Before submitting:

1. Ensure project builds successfully.
2. Regenerate docs when API comments or signatures change.
3. Confirm generated docs include your new or updated symbols.
