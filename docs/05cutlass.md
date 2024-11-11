# Layout
A Layout is a tuple of (`Shape`, `Stride`). Semantically, it implements a mapping from any coordinate within the `Shape` to an index via the `Stride`.

A Layout is a pair of `IntTuple`s: the `Shape` and the `Stride`. The first element defines the abstract shape of the Layout, and the second element defines the strides, which map from coordinates within the shape to the index space.
