# Loop Classification Protocol

Version 2026.10.07. If a rule changes, update the version and re-classify every labeled loop.

A loop is `convertible` if it passes all five checks below. The checks test whether a loop can be written as a
sequential stream. They aim to rule out any change in behavior but they are not a proof of equivalence. If a check
fails, the verdict is `not`, and `failed_checks` lists every failed check, separated by a semicolon. Checks other
than `source` assume that the loop has a source, so if a loop fails the `source` check, `failed_checks` should list
only `source`.

| Check             | Passes when                                                                      | Example stream form                                          |
|-------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------|
| `source`          | The loop draws its elements from an array, `Iterable`, `Iterator` or index range | `Arrays.stream(a)`, `list.stream()`, `IntStream.range(0, n)` |
| `exceptions`      | No checked exception can propagate out of the loop                               | —                                                            |
| `local_state`     | The loop assigns no variable declared outside it, except a single accumulator    | `sum`, `reduce`, `collect`                                   |
| `control_flow`    | Every exit from the loop has the shape of a short-circuit operation              | `anyMatch`, `allMatch`, `noneMatch`, `findFirst`, `limit`    |
| `iteration_order` | No iteration depends on another                                                  | —                                                            |

## Terminology

- **Loop's text** is the source code of a loop, excluding the bodies of the methods it calls and the lambda expressions
  and classes declared in it.
- **Loop's source** is what the loop draws its elements from, such as an array, an `Iterable`, an `Iterator` or an index
  range.
- **The current element** is a single element that an iteration draws from the source: `e` in `for (T e : xs)`, the
  result of `it.next()`, or the index of an index range. Elements read through an index, such as
  `a[i]` or `list.get(i)`, are ordinary reads.
- **A conjunct** is an operand of `&&` or `&`. **The loop condition's conjuncts** are the conditions that its top-level
  `&&` and `&` operators join. For instance, in `while (!found && n < max && it.hasNext())`, they are `!found`,
  `n < max` and `it.hasNext()`. If the loop condition has no top-level `&&` or `&`, the rules for its conjuncts apply to
  the whole condition.
- **An exit** is any way the loop can end other than its source's own test (e.g., `it.hasNext()`) becoming false:
    - a `break` statement that targets the loop or an outer loop;
    - a `continue` statement that targets an outer loop;
    - a `return` statement in the loop's text;
    - an exception that propagates out of the loop;
    - a conjunct of the loop condition other than the source's own test.
- **An exit condition** is the condition under which the loop takes an exit, such as `found` in
  `while (!found && it.hasNext())` or `e == target` in `if (e == target) break;`.
- **A carried value** is a value that an iteration reads in the loop's text and that an earlier iteration may have
  changed. The earlier change can happen in the loop's text or inside a method the loop calls. The read can be of a
  variable, field, array element, or the result of a method call, such as `seen.contains(x)` after `seen.add(y)`.
  Advancing a source that passes the `source` check (`it.next()` or the index update) does not carry values. An index
  that fails the `source` check, such as one that moves by a varying amount, is a carried value.
- **An accumulator** is a variable declared outside the loop whose updates fit a `reduce` or `collect`. An update can
  assign the variable, assign an element of the array it refers to, or call a method on the object it refers to.
  Examples include `sum += e`, `max = Math.max(max, e)`, `if (e > max) max = e`, and `map.merge(k, 1, Integer::sum)`.
  The update need not be associative, as in `h = 31 * h + e`, because sequential streams apply it in order. An object
  or array is an accumulator only if the loop uses it as a mutable result container, as in `list.add(e)` or the
  histogram `a[k] += e`, and code after the loop reads it. Other method calls, such as `writer.write(e)`, and stores at
  the index of an index range, such as `out[i] = f(e)` or `out[i] = f(out[i])`, are side effects, not reductions.
- **A limit counter** is a variable declared outside the loop that maps to `Stream.limit`. It has the following
  properties:
    - it is set to a constant before the loop;
    - it is incremented by 1 at most once per iteration;
    - every iteration that does not increment it either ends with a `continue` statement that targets the loop or
      takes an exit;
    - an exit condition compares it with a bound that does not change during the loop, as in `n < max` in the condition
      or `if (n >= max) break;`;
    - the loop reads it nowhere else.
- **A state change** is an assignment to a field, an array element or a variable declared outside the loop, or a call
  to a method that makes one. Logging, class loading and caches that only memoize a result are not considered state
  changes. Judge a call by the body of the method it calls, and the calls in that body by their documentation or by a
  name that clearly changes state, such as `add`, `put`, `set…`, `write` or `remove`.

## Checks

### `source`

A loop passes the `source` check if it matches one of the following patterns:

- an enhanced `for` loop over an array or `Iterable` object;
- a `for`, `while` or `do-while` loop whose condition has `it.hasNext()` as one of its conjuncts for a
  `java.util.Iterator`, and whose body calls `it.next()` exactly once on every path that goes back to the loop
  condition and never calls `it.hasNext()`;
- a `for`, `while` or `do-while` loop over an index range. The index is an `int` or a `long`. It is set once before the
  first iteration and compared in the condition, by `<`, `<=`, `>` or `>=`, with bounds that do not change during the
  loop. It moves toward the bound by a nonzero step that does not change during the loop, exactly once per iteration, in
  the `for` update or as the last statement of a `while` or `do-while` loop. Nothing else in the loop assigns it.

A `do-while` loop also needs the code before it to ensure that the first iteration has an element (`it.hasNext()` is
true or the index is within its bounds). Otherwise, it fails the `source` check.

A call to a method of a `java.util.Iterator` that `next()` delegates to, such as `nextChar()` in fop's `CharIterator`,
counts as a call to `it.next()`.

Every other loop fails. Examples include an `Enumeration`, linked-node traversal (`n = n.next`), `while (true)`, reading
from a stream, reader or queue until it is empty, and an iteration that draws more than one element.

### `exceptions`

A loop fails the `exceptions` check if a checked exception can propagate out of it.

The check follows the Java compiler's rules. A checked exception can propagate if a `throw` statement in the loop's text
throws it, or if the `throws` clause of a method or constructor that the loop's text calls declares it. Catching it
inside the loop stops it, but a `throw` statement in the `catch` block can throw a checked exception again.

### `local_state`

A loop fails the `local_state` check if it has more than one accumulator, or if its text assigns any of the following:

- a local variable or parameter declared outside the loop, other than an accumulator;
- a field that the loop's text also reads, other than an accumulator.

Assigning a limit counter or the index of an index range passes this check. If code after the loop reads one, it counts
as an accumulator.

A local variable declared outside the loop counts as declared inside it if every iteration assigns it before reading it
and no code after the loop reads the value it holds when the loop ends.

In a loop without an accumulator, assigning a field that the loop's text never reads passes, and so does any assignment
inside a method that the loop calls. If a later iteration reads an object that the loop changes, the `iteration_order`
check catches it.

A loop without an accumulator and without an `anyMatch`, `allMatch`, `noneMatch` or `findFirst` exit maps to
`forEachOrdered`, which keeps the order of the source's elements and is the stream operation meant for side effects. A
loop with an accumulator maps to `reduce` or `collect`, which expect functions without side effects, so it fails this
check if it makes any state change other than updating the accumulator, the index of an index range or a limit counter.

### `control_flow`

The `control_flow` check judges only the shape of a loop's exits. What an exit condition or an exit's result reads is
judged by the `iteration_order` check. For example, an exit condition that reads a flag that a method the loop calls can
set passes `control_flow` and fails `iteration_order`.

A `continue` that targets the loop passes `control_flow` because it maps to `filter`. An unchecked exception that
propagates out of the loop also passes this check because it propagates out of a stream the same way. A checked
exception that propagates out of the loop passes `control_flow` but fails the `exceptions` check.

A conjunct of the loop condition that reads neither the current element nor the accumulator passes. Whether what it
reads is a carried value is judged by the `iteration_order` check. Every other exit, including a conjunct that reads the
accumulator, must take one of the following forms:

- **`limit`:** the exit condition compares a limit counter with its bound.
- **`anyMatch`, `allMatch` or `noneMatch`:** the exit returns a constant `boolean`, or stores one in the accumulator.
- **`findFirst`, followed by `map` or `ifPresent`:** the exit returns, stores in the accumulator or uses only the
  matching element, or an expression of it.

An `anyMatch`, `allMatch`, `noneMatch` or `findFirst` exit has one more requirement. On every path through an iteration
that does not take the exit, the loop may only:

- compute the values that its exit conditions use, through declarations, assignments to variables other than the
  accumulator and method calls whose results those conditions use and that make no state change;
- take a `continue` that targets the loop.

Any other statement on such a path fails this check. Examples are a call made for its side effect, an update of the
accumulator, an assignment to a field or array element, and processing the element before or after evaluating the exit
condition. Processing/dropping elements until one matches a condition is the shape of `takeWhile`/`dropWhile`, which
this protocol does not accept.

A conjunct of the loop condition that becomes false only because the body set the accumulator, such as `!found` in
`while (!found && it.hasNext())`, forms one exit together with the assignments that set it. Reading the accumulator in
that conjunct is allowed.

### `iteration_order`

A loop fails the `iteration_order` check if any of the following conditions hold:

1. it adds elements to or removes elements from the source, such as with `add`, `remove` or `it.remove()`;
2. it writes to an element of the source that a later iteration reads;
3. it reads a carried value except for:
    1. an accumulator in its own updates, even if the loop has multiple accumulators;
    2. the accumulator in a conjunct of the loop condition that is part of an exit, such as `!found`;
    3. a limit counter in its increment and its exit condition.

## Applying the Checks

### Nested Loops

Each loop gets its own row in the worksheet and is classified on its own. When you classify the outer loop, the inner
loop is just code in the outer loop's body, and every check applies to that code. The inner loop's own source and its
exits that target only itself do not count against the outer loop.

### Uncertain Cases

Judge a called method by its body. If its body is unavailable, as for a native method, judge it by its documentation. If
a call can dispatch to several implementations, judge every implementation in the benchmark's sources and libraries.

Treat two references as different objects unless the code shows they are the same.

If a check still cannot be decided, the loop fails it, and the `notes` column records why.

### Performance

Performance is not a criterion. Boxing cost is a caveat in the write-up, not a reason to mark a loop `not`.

## Relation to LambdaFicator

The checks adapt the preconditions of LambdaFicator (Gyori et al., 2013) and deliberately widen them. Only stream
operations available in Java 8 count because some DaCapo benchmarks target Java 8. The checks differ from LambdaFicator
as follows:

- `source`: LambdaFicator accepts only enhanced `for` loops over a `Collection` (P1). This protocol also accepts
  `Iterable` objects, arrays, `Iterator` loops and index ranges.
- `exceptions`: no difference. This is LambdaFicator's precondition P2.
- `local_state`: LambdaFicator accepts one outer variable only as a compound assignment or increment (P3). This protocol
  accepts any single accumulator that fits a `reduce` or `collect`.
- `control_flow`: LambdaFicator rejects every `break` (P4) and allows one `return` only as `anyMatch` or `noneMatch`
  (P5). This protocol accepts any exit that maps to `anyMatch`, `allMatch`, `noneMatch`, `findFirst` or `limit`. It does
  not accept `takeWhile` and `dropWhile`, which need Java 9.
- `iteration_order`: LambdaFicator has no such check.

Therefore, results are not directly comparable with the applicability figures of LambdaFicator.

## References

Gyori, Franklin, Dig and Lahoda,
[Crossing the Gap from Imperative to Functional Programming
through Refactoring](https://dig.cs.illinois.edu/papers/lambdaRefactoring.pdf)
(ESEC/FSE 2013).
