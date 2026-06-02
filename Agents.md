# AGENTS.md

## Purpose

This file defines the engineering principles that any AI agent, coding assistant, or contributor must follow when designing, modifying, reviewing, and implementing code in this repository.

The goal is to produce code that is maintainable, extensible, testable, understandable, and resilient to change.

---

## Core Design Principles

### 1. Single Responsibility Principle, SRP

Each class, module, function, or component should have one clear reason to change.

When implementing code:

- Keep each unit focused on a single, well-defined responsibility.
- Avoid mixing unrelated concerns such as validation, persistence, formatting, networking, and business logic in the same place.
- Split large classes or modules when they begin to handle multiple independent responsibilities.
- Prefer small, focused functions over broad, multipurpose ones.

Before adding code, ask:

> Is this responsibility already handled somewhere else, or should this be a separate module?

---

### 2. Open-Closed Principle, OCP

Software components should be open for extension but closed for modification.

When implementing code:

- Design systems so new behavior can be added without rewriting stable, working logic.
- Prefer extension points such as interfaces, abstractions, configuration, composition, plugins, strategies, or handlers.
- Avoid editing core logic repeatedly for every new case when a polymorphic, registry-based, or strategy-based design would be cleaner.
- Keep existing behavior stable unless the task explicitly requires changing it.

Before modifying existing code, ask:

> Can this new behavior be added through extension rather than changing established logic?

---

### 3. Liskov Substitution Principle, LSP

Subtypes must be safely replaceable for their base types without breaking expected behavior.

When implementing code:

- Ensure subclasses, implementations, or derived components honor the contracts of their parents or interfaces.
- Do not weaken preconditions, strengthen postconditions unexpectedly, or throw surprising errors from a subtype.
- Avoid inheritance when the derived type does not truly behave like the parent type.
- Prefer composition over inheritance when behavior does not fit a strict “is-a” relationship.

Before creating a subtype, ask:

> Can this implementation be used anywhere the parent type is expected without surprising the caller?

---

### 4. Interface Segregation Principle, ISP

Code should not depend on methods, fields, or interfaces it does not use.

When implementing code:

- Prefer small, focused interfaces over large, general-purpose ones.
- Split “fat” interfaces into role-specific contracts.
- Do not force consumers to implement unused methods.
- Accept the narrowest interface required by a function or class.

Before adding to an interface, ask:

> Do all consumers need this method, or should this belong in a smaller, separate interface?

---

### 5. Dependency Inversion and Dependency Injection

High-level business logic should not depend directly on low-level infrastructure details. Both should depend on abstractions.

When implementing code:

- Keep business rules independent from databases, frameworks, file systems, HTTP clients, queues, and third-party services.
- Inject dependencies from the outside instead of hard-coding them inside classes or functions.
- Depend on abstractions, protocols, interfaces, or clearly defined contracts where useful.
- Make dependencies explicit through constructors, function parameters, or configuration.
- Avoid hidden global state and implicit service lookups unless they are already established project conventions.

Before adding a dependency, ask:

> Can this be supplied from the outside so the code remains testable and replaceable?

---

### 6. Loose Coupling

Modules should be as independent as possible and should share only the minimum information needed to collaborate.

When implementing code:

- Minimize direct knowledge between modules.
- Avoid making one component depend on another component’s internal structure.
- Prefer communication through stable interfaces, events, messages, DTOs, or well-defined APIs.
- Avoid unnecessary bidirectional dependencies.
- Keep changes localized whenever possible.

Before connecting modules, ask:

> Does this component really need to know about that component, or can they communicate through a smaller contract?

---

### 7. High Cohesion

The contents of a module, class, or function should be strongly related and focused on a unified purpose.

When implementing code:

- Group related behavior together.
- Keep unrelated behavior separate, even if it is convenient to place it in an existing file.
- Avoid utility dumping grounds that collect unrelated helpers.
- Name modules according to their cohesive purpose.
- Ensure public methods on a class support the same conceptual responsibility.

Before placing code in a module, ask:

> Does this code naturally belong here, or is this module becoming a miscellaneous collection?

---

### 8. Information Hiding

A module should expose a simple, stable interface while hiding internal implementation details.

When implementing code:

- Keep internal data structures, algorithms, and implementation details private whenever possible.
- Expose only what callers need.
- Avoid leaking database schemas, third-party response shapes, internal state, or framework-specific details across boundaries.
- Use clear public APIs and protect callers from unnecessary internal changes.
- Encapsulate complex operations behind intention-revealing methods.

Before exposing something publicly, ask:

> Does external code genuinely need this detail, or should it remain hidden behind an interface?

---

### 9. Law of Demeter, Principle of Least Knowledge

A component should communicate only with its immediate collaborators and should avoid reaching through chains of objects.

When implementing code:

- Avoid long chains such as `a.getB().getC().doSomething()`.
- Do not make callers navigate deep object graphs to perform simple tasks.
- Add intention-revealing methods to the appropriate owning object instead of exposing internals.
- Keep object relationships simple and local.
- Avoid coupling code to the hidden structure of other components.

Before using a call chain, ask:

> Am I talking to an immediate collaborator, or am I depending on the internals of a stranger?

---

### 10. Don’t Repeat Yourself, DRY

Every piece of logic, configuration, or business rule should have a single authoritative location.

When implementing code:

- Avoid copy-pasting logic across files, classes, tests, or configuration.
- Extract repeated business rules into shared functions, services, policies, constants, or configuration.
- Keep duplicated test setup under control with builders, fixtures, or helpers when appropriate.
- Do not over-abstract too early; remove duplication when the repeated code represents the same concept or rule.
- Ensure bug fixes and rule changes can be made in one place.

Before duplicating code, ask:

> Is this the same rule or concept repeated, and should it have one source of truth?

---

## Implementation Guidelines for Agents

When designing or changing code, agents must follow this workflow:

1. Understand the existing architecture before making changes.
2. Identify the smallest safe change that satisfies the request.
3. Preserve existing public behavior unless the task explicitly requires changing it.
4. Prefer clear, simple designs over clever abstractions.
5. Introduce abstractions only when they reduce coupling, remove meaningful duplication, or support real extension needs.
6. Keep business logic separate from infrastructure and presentation concerns.
7. Make dependencies explicit and easy to replace in tests.
8. Avoid broad rewrites unless the existing design prevents a correct or maintainable solution.
9. Add or update tests for changed behavior when the project has a testing pattern.
10. Document non-obvious design decisions in code comments or project documentation.
11. Ask for any explanation or clarification from the user before implementing anything until a shared understanding is reached.
