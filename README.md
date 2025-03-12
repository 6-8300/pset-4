# 6.8300 Problem Set 4

> Differentiable Rendering

This problem set will cover

- Neural networks as representations of signals
- Sphere tracing and differentiable volume rendering
- Neural radiance fields

Please refer to the READMEs in `problem1/`, `problem2/`, `problem3/` for implementation details and submission instructions.

## Grading

### Points breakdown
- Problem 1
    - Multiple choice: 10%
    - Implementing SIRENs: 20%
- Problem 2 [Total of 40 points]
    - Sphere tracing: 20%
    - Volumetric rendering: 20%
- Problem 3 [Total of 40 points]
    - Implementation and understanding of the rendering: 30%
    - NeRF results images: 10%

### Grading mechanism

For each problem, look in the description to see how its graded. The grades are all in `[0, 1]`. We take a weighted average with the weights above to get the final grade in terms of all the problems:

```python
import numpy as np
weights_names = ["p1_mc", "p1_siren", "p2_sphere", "p2_vol", "p3_tbd1", "p3_tbd2"]
weights = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2])          # weights for each problem
grades = np.array([0.90, 0.85, 0.80, 0.75, 0.90, 0.85])     # example grades in same order - this is determined by the grader per-problem
final_score = np.dot(weights, grades).item()                # compute weighted sum
print(f"Final grade: {final_score:.4%}")
```