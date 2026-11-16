# Product explorer

The app in which a product author builds and checks a product model and a researcher inspects one: exploring its product states, viewing its sequence graph, saving it for the gym. Replaces the legacy DASG app; running the cell is not its job.

## Language

### Roles

**Product author**:
The person who builds a product model, checks it by exploring its product states and saves it for the gym.
_Avoid_: user, operator, designer

**Researcher**:
The person who opens an existing product model or sequence graph and inspects it without editing it.
_Avoid_: viewer, analyst

**Product explorer**:
The app the product author and the researcher work in.
_Avoid_: DASG app, HMI, GUI, editor, workbench

### Authoring

**Product setup**:
What exploring needs about a product beside the product model: its base parts and target parts, and optionally its stuck-part rules (see the decision-making context; the explorer keeps them when it stores the setup). Authored by the product author and kept in the product folder. The tools and timing settings belong to the cell, not to the product setup.
_Avoid_: env config, settings, scenario

**Product folder**:
The one folder that holds everything of one product model: the product model file, its part meshes, its contact data and its product setup. Moving or sharing the folder moves the product. Every tool finds the product model file by one **loader rule** (the product-model type): the path a graph export's provenance names, else the single XML at the folder root, else (only when the root has none) the single XML in the folder's `xml` subfolder; anything else is an error naming the folder and the XML files found. The product setup stays at the folder root.
_Avoid_: project, workspace, data dir

**Hand-made connection**:
A screw, safety ring or fixed connection the product author adds, edits or removes in the product explorer, each as one undoable edit. A screw connection is saved with the screw as its first part and a screw head the screwdrivers of the cell in use support. Geometric connections are never hand-made: they are deleted, or detected again for one part pair (the connections of that pair are replaced, nothing else changes); detecting the whole product happens only on import.
_Avoid_: manual connection, custom connection

**3D files repository**:
The shared repository of CAD files, meshes and hand-made URDFs of the products and the cell equipment of the Transformer cell (`crc_1574_3D_Files`). A product's product folder lives inside that product's folder there; only test products stay with the code ([ADR-0011](../../adr/0011-product-folders-live-in-the-3d-files-repository.md)).
_Avoid_: product library, product repo, dataset

**Test product**:
A small product folder kept with the code so tests can run without the 3D files repository. It may be a frozen copy of a real product; it is never the source of a benchmark product.
_Avoid_: fixture product, sample product

**Cell file**:
The optional file in the existing cell file format whose screwdrivers and grippers are the tools of the available actions, and whose travel-time table (the file next to it) replaces the default settings. Without one the default cell is used and the session view says so. Its fingerprint, which covers the travel-time table, is recorded with a sequence graph, so a graph built with one cell file is stale for another, and for the same cell file with changed times.
_Avoid_: cell config, cell settings file

**Mesh folder import**:
Turning a folder of STL or OBJ part meshes into a new product model in one step. Contact detection runs in the background and can be cancelled; meshes are taken as given (millimetres, one product frame). The result lives in memory with an empty product setup until its first save writes the product folder.
_Avoid_: load, open, convert

**Mesh update**:
Taking the meshes of a new CAD export folder into the open product, in one undoable edit that ends with a report. Parts are matched to the meshes by file name. A part with a changed mesh gets the new mesh and its geometric connections with its neighbours detected again (hand-made connections, renames and properties stay); a part with no mesh file is removed; new mesh files join only when the product author ticks them. Merged subassemblies are skipped and listed as "re-merge by hand": the explorer knows them only from the session in which they were merged, because no merge record is kept in the product folder. The new export folder is never written.
_Avoid_: re-import, refresh, sync

### Export

**Export**:
A file the product explorer derives from a product folder for a consumer: a sequence graph file or per-part URDFs. It can always be rebuilt, so it is never kept in the 3D files repository; a consumer that must keep one stores a copy with its own results.
_Avoid_: output, artifact, save

**Product root**:
The absolute folder under which the product folders live on this machine, for example the checkout of the 3D files repository. It comes from the app setting (a JSON file in the user's configuration directory, set in the Export menu), else the environment variable `PRODUCT_EXPLORER_PRODUCT_ROOT`, else the root of the 3D files repository that holds the product folder. A recorded path is relative to the product root (base `product_root`) or to the nearest higher functional root, the product root's ancestors up to the repository root (base `product_root+N`), or absolute when it lies under none (base `absolute`). The product root itself is never written into an export.

**Provenance**:
What an export records about its origin: the product, where its product folder and cell file sit (with a **Product root** in effect: relative to it, or to a higher functional root, or the absolute path when they lie under none; with none: from the root of the 3D files repository when they lie in one, otherwise relative to the export folder, never an absolute path; the repository commit is information only), fingerprints of the product model, product setup and cell file, the extraction depth (down to the disassembled states or a full teardown, chosen when the graph is generated; see [Sequence graph](../sequence-graph/CONTEXT.md)), and the code version. An export whose fingerprints differ from the product folder belongs to an older product model.
_Avoid_: metadata, header info

### Exploring

**Exploring**:
Going from product state to product state by choosing available actions by hand. Jumping to a node of the whole sequence graph is not exploring.
_Avoid_: steering, stepping, playing, simulating

**Available action**:
An action the POMDP's own capabilities accept in the current product state.
_Avoid_: valid action, possible move

**Rejection reason**:
Why a chosen action is a **Rejected action** (see decision making): a restriction, a connection that still holds, or a failed feasibility check.
_Avoid_: blocked, why-blocked, invalid action

**Edit history**:
The product author's edits to the product model and product setup, and the choice of the cell file, in one session, which undo and redo step through. Opening a product clears it. After any edit, undo or redo the sequence graph is stale and the available actions of the current product state are re-checked on the edited model.
_Avoid_: undo stack, change log

**Stale sequence graph**:
A sequence graph extracted before the latest edit to the product model or a cell file other than the current one; it waits for a rebuild.
_Avoid_: outdated graph, dirty graph
