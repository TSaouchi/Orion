<p align="center">
  <strong style="font-size: 24px; color: red;">🚧 Work in Progress 🚧</strong>
</p>

<p align="center">
  <img src="Documentation/source/_static/logo/orion_logo.png" alt="Orion" width="200" />
</p>

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache License, Version 2.0.")](LICENSE)

# About
This repository provides a Python-based API for post-processing large-scale data. The API is designed to handle multi-dimensional data across various zones, time instants, and variables, offering scalable and parallelized computation capabilities.

## Table of Contents

- [Features](#features)
- [Dataset structure](#Dataset-structure)
- [Installation](#installation)
- [Documentation](#Documentation)
- [Usage](#usage)
  - [Basic Example](#basic-example)

## Features

- **Organized Data Structure**: Manage multi-dimensional data across zones (e.g., physical regions), instants (e.g., time steps), and variables (e.g., physical quantities like velocity or pressure).
- **Efficient Data Handling**: Relies on [HDF5](https://docs.alliancecan.ca/wiki/HDF5/fr) for efficient storage and access to multi-dimensional data, ensuring high performance and flexibility.
- **Efficient Computation**: Uses [Dask](https://www.dask.org/) arrays to handle large datasets lazily, triggering computation only when needed.
- **Parallelization**: Perform computations across zones and instants using **ThreadPoolExecutor** for multi-threaded processing.
- **Custom Metadata**: Attach custom attributes (metadata) to zones, instants, or variables for easy categorization and filtering.
- **Expression-Based Computation**: Dynamically compute new variables using literal expressions (e.g., `"new_var = var1 * var2"`).

## Dataset structure

Orion's data structure is heavily inspired by the hierarchical data organization of both [HDF5](https://docs.alliancecan.ca/wiki/HDF5/fr) and [CGNS](https://cgns.github.io/)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0a2933', 'secondaryColor': '#00ccff', 'tertiaryColor': '#ffd700', 'mainBkg': '#0a2933', 'nodeBorder': '#00ccff', 'clusterBkg': '#0a293366', 'clusterBorder': '#00ccff', 'lineColor': '#ffd700', 'fontFamily': 'arial', 'fontSize': '14px'}}}%%

graph TD
    C[Base Custom Attributes] --> A[Base]
    A --> B[Zones OrderedDict]
    G[Zone Custom Attributes] --> D[Zone1 Zones object]
    B --> D
    B -.-> E[More Zones...]
    K[Instant Custom Attributes] --> H[Instant1 Instants object]
    D --> F[Instants OrderedDict]
    F --> H
    F -.-> I[More Instants...]
    H --> J[Variables OrderedDict]
    J --> L[Variable1 Variables object]
    J --> M[Variable2 Variables object]
    J --> N[Variable3 Variables object]
    J -.-> O[More Variables...]
    L --- P[Variable Custom Attributes]
    M --- Q[Variable Custom Attributes]
    N --- R[Variable Custom Attributes]

    classDef default fill:#0a2933,stroke:#00ccff,color:#ffffff;
    classDef extensible fill:#0a2933,stroke:#ffd700,color:#ffd700,stroke-dasharray: 5 5;
    classDef customAttr fill:#00ccff,stroke:#0a2933,color:#0a2933;
    class E,I,O extensible;
    class C,G,K,P,Q,R customAttr;
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/TSaouchi/Orion.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Orion
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Documentation 

The project documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/) and can be accessed via the **Read_The_Docs.html** portal.

## Usage

### Basic Example

[Example](Documentation\source\_static\pictures\Basic_Example.ipynb)

# Author
The Orion library was created and is maintained - in my free time ;) - by Toufik Saouchi
