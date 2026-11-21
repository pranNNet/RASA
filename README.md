# RASA
resolution-aware stencil attention nerds

- Two primary mechanism under investigation
    - A surrogate model that uses AMR (automatic mesh refinement) techniques to refine and coarsen domain being modeled
        - Goal is to minimize computational effort and memory on unimportant regions
            - Ideally scalable to a 3D problem
    - Using attention as a local operator
        - Goal is to develop a robust kernel that avoids smoothing and adaptable to the AMR-esque grid
        - local operator learns self-similar physics across resolutions?
            - maybe it just learns numerics instead (numerics are not as self-similar as the physics are)

Notes:
1. Domain-wise, the finest grid level is the grid size of the training dataset
2. dunno kinda unsure what i be doin yo