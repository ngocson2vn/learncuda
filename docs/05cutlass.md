# Layout
A Layout is a tuple of (`Shape`, `Stride`). Semantically, it implements a mapping from any coordinate within the `Shape` to an index via the `Stride`.

A Layout is a pair of `IntTuple`s: the `Shape` and the `Stride`. The first element defines the abstract shape of the Layout, and the second element defines the strides, which map from coordinates within the shape to the index space.

In NVIDIA **CuTe** (C++ templates for tensor computation), the concepts of **coordinate space** and **index space** are fundamental to understanding how data is addressed and manipulated in tensor operations. Here's a breakdown of the differences:

---

### **1. Coordinate Space**
- **Definition**: Represents the logical, multi-dimensional structure of a tensor.
- **Description**:
  - A tensor is typically conceptualized as a multi-dimensional array, and its elements are identified by their coordinates (e.g., `(x, y, z)` in 3D space).
  - Coordinates are multi-dimensional and correspond to the logical organization of the tensor data.
  - This space aligns with the **logical view of the data layout**.
  - Each coordinate in the coordinate space is independent of the memory layout.

- **Examples**:
  - In a 2D tensor with dimensions \( 4 \times 3 \), coordinates might include `(0, 0)`, `(1, 2)`, etc.
  - Logical access to tensor elements (row, column, etc.) occurs in the coordinate space.

---

### **2. Index Space**
- **Definition**: Represents the physical, one-dimensional linear layout of tensor data in memory.
- **Description**:
  - Tensors are ultimately stored in contiguous memory, and each element is accessed via an index in this linear space.
  - The mapping from coordinates (logical space) to indices (linear memory space) is determined by the tensor's **layout**.
  - This space corresponds to the **physical view of the data layout**.
  - Index space depends on the memory layout scheme (e.g., row-major, column-major, or more complex tiling layouts).

- **Examples**:
  - In a row-major layout of a \( 4 \times 3 \) tensor, the coordinate `(1, 2)` maps to the linear index `5` (assuming zero-based indexing and row-major order).

---

### **Key Differences**
| Aspect              | Coordinate Space                                | Index Space                        |
|---------------------|-------------------------------------------------|------------------------------------|
| **Nature**          | Logical, multi-dimensional                      | Physical, one-dimensional          |
| **Purpose**         | Describes positions within the tensor logically | Describes positions in memory      |
| **Example**         | `(2, 3, 4)` in a 3D tensor                      | `17` (linear index) in memory      |
| **Mapping**         | Multi-dimensional coordinates                   | Flattened, memory-specific indices |
| **Impact of Layout**| Layout-independent                              | Layout-dependent                   |

---

### **Relationship Between the Two**
- The mapping between coordinate space and index space is handled by the tensor's **layout policy**.
- For example:
  - **Row-major layout**: Indices increase sequentially along rows.
  - **Column-major layout**: Indices increase sequentially along columns.
  - **Tiled layouts** (e.g., for GPUs): Indices are computed based on a hierarchical tiling structure for efficient memory access.

Understanding these spaces is crucial when implementing or optimizing tensor operations, as transformations between coordinate space and index space affect both **logical correctness** and **performance**.

# CuTe Layout Algebra
In computer science, **functional composition** refers to the process of combining two or more functions to produce a new function. The output of one function becomes the input of the next. This concept is rooted in mathematics and is widely used in programming, especially in functional programming paradigms.

### Notation in Mathematics
In mathematical terms, if you have two functions $f$ and $g$:
- $ f: X \to Y $
- $ g: Y \to Z $

The composition of $ f $ and $ g $, denoted as $ g \circ f $, is a new function $ h $:
- $ h(x) = g(f(x)) $
- $ h: X \to Z $

### Functional Composition in Programming
In programming, functional composition allows for chaining operations or transforming data through a series of functions. For example:

#### Example in Python:
```python
def double(x):
    return x * 2

def increment(x):
    return x + 1

# Compose the functions
def composed_function(x):
    return increment(double(x))

result = composed_function(3)  # Output: 7
```

Here, `composed_function` applies `double` first and then `increment`.

#### Functional Programming Paradigm
Many functional programming languages, like Haskell, Scala, or JavaScript, support composition directly using built-in utilities:

- **JavaScript Example**:
```javascript
const double = x => x * 2;
const increment = x => x + 1;

const composedFunction = x => increment(double(x));
console.log(composedFunction(3)); // Output: 7
```

- Using utility libraries like Lodash:
```javascript
import { flow } from 'lodash';

const composedFunction = flow([double, increment]);
console.log(composedFunction(3)); // Output: 7
```

### Benefits of Functional Composition
- **Code Reusability**: Small, single-purpose functions can be reused in multiple compositions.
- **Modularity**: Functions can be composed to create complex operations in a clear and maintainable way.
- **Declarative Style**: Composition often leads to more readable code that focuses on *what* should be done rather than *how* it’s done.

### Common Use Cases
1. **Data Processing Pipelines**: Transform data through a series of steps.
2. **Middleware in Web Frameworks**: Chain operations like logging, authentication, and request handling.
3. **Signal Processing**: Apply transformations and filters to signals.
4. **Machine Learning Pipelines**: Combine feature extraction, data preprocessing, and model inference.


# WGMMA
In the CUTLASS paradigm for MMA, the `cute::gemm` method is designed to expose architecture-specific MMA instructions via a uniform interface. (Indeed, if you examine the SM80 tutorial GEMM kernel, you’ll see that the `cute::gemm` call there is syntactically identical to that given above.) However, the definitions of the arguments involved in the cute::gemm call involve many WGMMA-specific aspects:

- The definition of the `TiledMMA` object `tiled_mma` encapsulates the information needed for `cute::gemm` to dispatch to a specific `wgmma` PTX instruction.
- The layouts of the SMEM tensors `sA` and `sB` must be defined to be compatible with `wgmma`.
- The fragments `tCrA`, `tCrB`, and `tCrC` are constructed as thread-level partitions of the data using the `TiledMMA` object, and as such have WGMMA-specific layouts that the programmer should be aware of.
- The fragments `tCrA` (if sourcing operand `A` from SMEM) and `tCrB` aren't register-backed tensors whose values are copied from SMEM, but rather matrix descriptors constructed on top of SMEM.

## MMA Atom: `MMA_64x64x16_F16F16F16_SS`
cutlass/include/cute/arch/mma_sm90_gmma.hpp
