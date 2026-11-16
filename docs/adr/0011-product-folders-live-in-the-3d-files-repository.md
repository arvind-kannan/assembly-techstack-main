# Product folders live in the 3D files repository

The planner-ready **product folder** of every real product (product model, part meshes, contact data, product setup) lives in the **3D files repository** (`crc_1574_3D_Files`), in `<Part>/product/` next to the part's `cad/ meshes/ urdf/ docs/`, so a product sits in one place with the CAD it comes from. Each product's author maintains its product folder. The techstack keeps only **test products**: small product folders the tests need, such as the bolt-and-nut benchmark case and a frozen copy of the EMO motor. The techstack reaches the 3D files repository through a sibling clone (sparse checkout); examples and benchmark runs take the product folder path explicitly, and each export pins the product version through its provenance (`repo_commit`). CI never needs the 3D files repository.

Decided in #182.

## Considered options

- **Product folders in the techstack, next to the code** (Git LFS, ADR-0001): one repo for CI and students, but the techstack grows with every benchmark product and the product model drifts from its CAD unnoticed. It also contradicted the glossary.
- **Git submodule of the 3D files repository**: the techstack would name the exact product version, but it adds a third mechanism next to the subtrees (ADR-0002), a plain clone pulls about 870 MB of LFS CAD zips, and every product update needs a bump commit.
- **Subtree or copy into the techstack**: keeps two homes of one product.

## Consequences

- The angle grinder, battery and EMO product folders move out of the techstack; examples and tools point to the 3D files repository path.
- A test product may drift from its real product; that is allowed, it is only test data.
- Exports stay out of the 3D files repository; a consumer that keeps one stores it with its results.
- The 3D files repository gets `*.npy` in its LFS rules and the `product/` folder in its README.
