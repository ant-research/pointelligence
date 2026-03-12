<h1 align="center">Pointelligence<br><sub><sup>Accelerating Point Cloud Learning for Spatial Intelligence</sup></sub></h1>

Pointelligence is a repository for 3D point cloud research, currently featuring:

* the official implementation of [PointCNN++](https://arxiv.org/abs/2511.23227) (CVPR 2026)—a significant next evolution of [PointCNN](https://github.com/yangyanli/PointCNN) (NeurIPS 2018).

# Clone

Clone the repository with third-party submodules (FCGF and Pointcept) recursively:

```shell
git clone --recursive https://github.com/ant-research/pointelligence.git
cd pointelligence
```

For reproducibility, checkout the following commits in the submodules:

```shell
# FCGF (examples/FCGF)
cd examples/FCGF && git checkout pointcnnpp-version && cd ../..

# Pointcept (examples/Pointcept)
cd examples/Pointcept && git checkout pointcnnpp-version && cd ../..
```

If you have already cloned without `--recursive`, run `git submodule update --init --recursive` to fetch the submodules.

# Installation

Some operators are implemented with C++/CUDA as PyTorch extensions, which could be built and installed with the following commands:

```shell
cd extensions
pip install --no-build-isolation -e .
```

# Basic Usages

## Point Cloud Registration Task

in `examples/FCGF`

## Point Cloud Segmentation Task

in `examples/Pointcept`

# Citation
Pointelligence is the repo for the official implementation of:
* [PointCNN++: Performant Convolution on Native Points](https://arxiv.org/abs/2511.23227)\
    [Lihan Li](https://lihhan.github.io/), Haofeng Zhong, Rui Bu, Mingchao Sun, [Wenzheng Chen](https://wenzhengchen.github.io/), [Baoquan Chen](https://baoquanchen.info/), [Yangyan Li](https://yangyan.li)
    ```text
    @misc{li2025pointcnnperformantconvolutionnative,
          title={PointCNN++: Performant Convolution on Native Points}, 
          author={Lihan Li and Haofeng Zhong and Rui Bu and Mingchao Sun and Wenzheng Chen and Baoquan Chen and Yangyan Li},
          year={2025},
          eprint={2511.23227},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2511.23227}, 
    }
    ```
  
# Feature Requests and Issues
To ensure they are tracked effectively, please submit feature requests and issue reports here rather than via email.


# Advanced Topics

While we provide a suite of ready-to-use backbones, our framework is explicitly designed to facilitate the construction of custom network architectures from scratch. We review several key concepts below; combined with the reference implementations in the `models` directory, these resources are intended to help users quickly master the library's workflow.

## Ragged Tensors for Efficient Batching

### The Inefficiency of Padding
The total number of points often varies significantly from one sample to another within a single batch. As illustrated below, the **straightforward approach** deals with this irregularity (e.g., samples having 2, 7, and 4 points) by forcing data into fixed-size dense tensors. While this satisfies the rigid structural requirements of standard frameworks, it overlooks the data's inherent sparsity. Smaller samples must be padded out with non-existent "ghost" data, squandering significant memory and compute cycles on empty space.

```text
Scenario: A batch of 3 irregular samples with 2, 7, and 4 points respectively.

Legend:
[P] = Valid Point/Feature Data
[.] = "Ghost Data" (Padding/Wasted Memory)


+-----------------------------------------------------------+
| THE STRAIGHTFORWARD APPROACH: Fixed-Size Dense Tensor     |
| Status: WASTEFUL. Forces data to match the largest dim.   |
+-----------------------------------------------------------+
To create a uniform grid, every sample must be padded to match
the largest necessary dimension (at least 7).

Batch Memory Layout (Fixed grid):
Row 0 (Sample 1):  [P][P][.][.][.][.][.]
Row 1 (Sample 2):  [P][P][P][P][P][P][P]
Row 2 (Sample 3):  [P][P][P][P][.][.][.]
                    ^Valid^      ^Wasted^

VISUAL RESULT: Significant portions of memory are useless padding.
```

### The Ragged Tensor Solution
Ragged tensors represent a **dedicated solution** to the inefficiency shown above. As visualized below, this format is explicitly designed to handle irregularity. Instead of maintaining separate, padded rows in a grid, the Ragged tensor stores the entire batch as a single, contiguous sequence containing only valid points. This packed data is managed by a lightweight auxiliary metadata structure that tracks the individual sample sizes. By eliminating padding entirely, this approach ensures processing applies only to actual geometric and feature data.

```text
+-----------------------------------------------------------+
| THE DEDICATED SOLUTION: Ragged Tensor                     |
| Status: EFFICIENT. Stores only what exists.               |
+-----------------------------------------------------------+

[A. Contiguous Data Storage]
Padding is eliminated. The entire batch is flattened into one
tightly packed sequence of valid data only:

Memory Layout:
[P][P] [P][P][P][P][P][P][P] [P][P][P][P]
\___/  \___________________/  \_________/
 S1(2)          S2(7)            S3(4)

VISUAL RESULT: Zero wasted space.


[B. Auxiliary Metadata]
A small separate structure tracks where samples sizes:

Sample Sizes:  [ 2,  7,  4 ]
```

## Downsampling and Upsampling

### Downsampling
We employ a voxel-based strategy using the $grid\_sample\_filter$ routine with $center\_nearest$ reduction mode. Instead of snapping coordinates to the grid, this method selects the point nearest to the voxel center, which preserves thin structures and sparse regions better than random downsampling while maintaining comparable negligible latency. To ensure efficiency, the routine processes batched samples as a single, unified point cloud, leveraging highly optimized GPU sorting and search primitives.

Because the $grid\_sample\_filter$ routine relies on the $grid\_size$ argument, the resolution of the processed point cloud is directly tied to this parameter. Conceptually, $grid\_size$ acts as the fundamental spatial unit, analogous to pixel size in the image domain. With this baseline established, architectural concepts such as $receptive\_field$ and $strides$ can be intuitively defined as relative multiples of $grid\_size$.

The $grid\_size$ parameter governs the critical trade-off between computational efficiency and model accuracy. A smaller $grid\_size$ enhances spatial resolution—potentially improving accuracy—but incurs greater computation and memory overhead. Conversely, a larger $grid\_size$ reduces resource demands but may sacrifice fine-grained detail. To mitigate issues from irregular data, such as excessive point concentrations in small regions, we recommend applying a preliminary 'sanity' downsample (e.g., at $\frac{1}{3} \times grid\_size$) before feeding the data into a network configured for the target $grid\_size$.

### Upsampling
Unlike in image processing where target pixel locations are fixed, the spatial locations of upsampled points in a point cloud are not inherently known. Therefore, a recommended practice is to explicitly reuse the retained set of original, pre-downsampled points as the upsampling target. This approach is both efficient and unambiguous. Note that the retention and retrieval of these original points must be managed by the enclosing pipeline or calling procedure.

## Neighborhood Computation and Representation

We opt for a fixed radius search over a fixed-number (K-Nearest Neighbors, or KNN) search, as its spatially-local receptive field is better suited for spatial learning, whereas KNN is often a choice imposed by architectural limitations. Given sets of "source" and "query" points—which may be identical—and a specified radius, the $radius\_search$ routine computes neighborhood points using highly optimized GPU sorting and search primitives. This routine supports batched point cloud samples by processing the entire batch as a single, unified point cloud, while ensuring that the search remains logically confined to each individual sample.

As illustrated in the vertical flow diagram below, the connectivity of a batched point cloud is most efficiently represented as a unified list of $(i, j)$ pairs, effectively "flattening" the complex topology of multiple samples into a compact stream of edges. In this scheme, every point in the batch is assigned a unique global index, allowing neighborhoods to be defined simply by the link between a query point $i$ and its neighbor $j$. This approach elegantly stores only the valid edges that actually exist. Crucially, no lines cross the empty gaps between Sample 1, Sample 2, and Sample 3.

```text
             [-SAMPLE 1-]   [-------------SAMPLE 2--------------]  [-----------SAMPLE 3-----------]
Query (i):     0    1        2  3      4      5      6    7    8    9            10    11    12  
               |    |        |  |      |      |      |    |    |    |            |     |     |   
               |-.  |-.      |  |-.-.  |-.-.  |-.-.  |-.  |-.  |    |--.--.--.   |--.  |--.  |--.
               | |  | |      |  | | |  | | |  | | |  | |  | |  |    |  |  |  |   |  |  |  |  |  |
               v v  v v      |  v v v  v v v  v v v  v v  v v  |    v  v  v  v   v  v  v  v  v  v
Neighbor (j):  0 1  0 1      2  3 4 5  3 4 5  3 4 5  6 7  6 7  8    9  10 11 12  9  10 9  11 9  12
   
The connections from the diagram above, flattened into two parallel arrays:
             [-SAMPLE 1-]   [-------------SAMPLE 2--------------]  [-----------SAMPLE 3-----------]
Query (i):     0 0  1 1      2  3 3 3  4 4 4  5 5 5  6 6  7 7  8    9  9  9  9   10 10 11 11 12 12
Neighbor (j):  0 1  0 1      2  3 4 5  3 4 5  3 4 5  6 7  6 7  8    9  10 11 12  9  10 9  11 9  12

+---------------------------------------------------------------------------------+
| THE UNIFIED LIST: (i, j) pairs                                                  |
| Status: EFFICIENT. A compact stream of all neighborhood edges in the batch.     |
+---------------------------------------------------------------------------------+
```

## Convolution on Native Points

The full process of convolution on native points involves four steps: output location generation, neighborhood search, convolution triplet construction and the Matrix-Vector Multiplication and Reduction (MVMR). The actual convolution computation occurs in the final MVMR stage, while the preceding three steps serve to structure the input data for this calculation.

### Output Location Generation
This step defines the spatial centers for the convolution operations. Depending on the desired architectural effect, the output locations are generated in one of three ways:
* Standard Convolution ($stride=1$): The input points serve directly as the output locations.
* Strided Convolution ($stride > 1$): Output locations are generated by downsampling the input points via $grid\_sample\_filter$ with a target grid size of $grid\_size_{input} \times stride$ (see [Downsampling](#downsampling)). Note that the $grid\_size$ of the output point cloud would be $grid\_size_{input} \times stride$.
* Transposed Convolution (Upconvolution): The output locations are explicitly set to the pre-calculated 'upsampled' points, as described in the [Upsampling](#upsampling) section.

### Neighborhood Search
This step executes the neighbor finding process detailed in [Neighborhood Computation and Representation](#neighborhood-computation-and-representation) by invoking the $radius\_search$ routine. The generated output locations serve as the **query points**, while the input points serve as the **source points**.

The search radius is determined by the formula $radius = radius\_scaler \times grid\_size$. The $radius\_scaler$ defines the geometric relationship between the spherical search region and the cubic receptive field (governed by the $receptive\_field$ hyperparameter). Common configurations for $radius\_scaler$ include:

* **Inscribed Sphere:** $radius\_scaler = \frac{1}{2} \times receptive\_field$. The search ball is the largest sphere that fits inside the ${receptive\_field}^3$ cube.
* **Equal Volume (Default):** $radius\_scaler = \sqrt[3]{\frac{3}{4 \pi}} \times receptive\_field$. The search ball has the same volume as the ${receptive\_field}^3$ cube.
* **Circumscribed Sphere:** $radius\_scaler = \frac{\sqrt{3}}{2} \times receptive\_field$. The search ball is the smallest sphere that encloses the ${receptive\_field}^3$ cube.

### Convolution Triplet Construction

To perform convolution on the irregular structure as depicted by the neighborhood $(i, j)$ pairs, we must explicitly link each neighborhood edge to the model's parameters. As illustrated in the diagram below, this is achieved by extending the neighborhood list into Convolution Triplets $(i, j, k)$. Unlike image convolutions where weight usage is implicit based on a fixed pixel grid, point clouds require a calculated assignment. As illustrated below, for every connection between a query point $i$ and neighbor $j$, a specific kernel weight index $k$ (from a shared bank of weights, e.g., $0 \dots 8$) is determined based on the position of neighbor $j$ within the local coordinate system centered at query point $i$. This "triplet" structure explicitly routes data from the neighbor $j$, through the correct weight $k$, to the target point $i$. Note that the kernel weights are shared across all samples in the batch. This triplet representation provides the basis for an efficient convolution implementation.

```text
             [-SAMPLE 1-]   [-------------SAMPLE 2--------------]  [-----------SAMPLE 3-----------]
Query (i):     0    1        2  3      4      5      6    7    8    9            10    11    12  
               |    |        |  |      |      |      |    |    |    |            |     |     |   
               |-.  |-.      |  |-.-.  |-.-.  |-.-.  |-.  |-.  |    |--.--.--.   |--.  |--.  |--.
               | |  | |      |  | | |  | | |  | | |  | |  | |  |    |  |  |  |   |  |  |  |  |  |
               v v  v v      |  v v v  v v v  v v v  v v  v v  |    v  v  v  v   v  v  v  v  v  v
Neighbor (j):  0 1  0 1      2  3 4 5  3 4 5  3 4 5  6 7  6 7  8    9  10 11 12  9  10 9  11 9  12
               | |  | |      |  | | |  | | |  | | |  | |  | |  |    |  |  |  |   |  |  |  |  |  |
               v v  v v      v  v v v  v v v  v v v  v v  v v  v    v  v  v  v   v  v  v  v  v  v
Kernel (k):    4 5  5 4      4  4 6 8  2 4 5  1 3 4  4 7  1 4  4    4  2  6  8   6  4  2  4  0  4

The connections from the diagram above, flattened into three parallel arrays:
             [-SAMPLE 1-]   [-------------SAMPLE 2--------------]  [-----------SAMPLE 3-----------]
Query (i):     0 0  1 1      2  3 3 3  4 4 4  5 5 5  6 6  7 7  8    9  9  9  9   10 10 11 11 12 12
Neighbor (j):  0 1  0 1      2  3 4 5  3 4 5  3 4 5  6 7  6 7  8    9  10 11 12  9  10 9  11 9  12
Kernel (k):    4 5  5 4      4  4 6 8  2 4 5  1 3 4  4 7  1 4  4    4  2  6  8   6  4  2  4  0  4
+---------------------------------------------------------------------------------+
| THE UNIFIED LIST: (i, j, k) triplets                                            |
| Status: EFFICIENT. A compact stream of all computation edges in the batch.      |
+---------------------------------------------------------------------------------+
```

More specifically, to determine the kernel index $k$, neighborhood points are first transformed into the query point's local coordinate system and normalized to a unit scale. A simple $voxelize\_3d$ routine then uniformly discretizes this unit space into $kernel\_size^3$ voxels; the index $k$ is assigned based on which voxel a neighbor occupies. Although the visualization above sorts triplets by query index $i$ for clarity, the actual implementation typically re-sorts them by kernel index $k$. This optimization reorganizes the data to maximize efficiency during the subsequent Matrix-Vector Multiplication and Reduction (MVMR) stage.

It is crucial to distinguish between $receptive\_field$ and $kernel\_size$, as these concepts are explicitly decoupled in our framework. While often synonymous in single-layer image convolutions, here they serve distinct roles: $receptive\_field$ defines the **physical spatial extent** of the neighborhood search, whereas $kernel\_size$ determines the **resolution** of the local voxelization (i.e., the granularity of the weight grid applied within that space). This decoupling provides flexible, fine-grained control over the architectural design.

### Matrix-Vector Multiplication and Reduction (MVMR)

As detailed in the research paper of [PointCNN++](https://arxiv.org/abs/2511.23227), the actual heavy lifting of the convolution arithmetic occurs in this final stage. To encapsulate this complexity, we provide a high-level $PointConv3d$ layer. The $forward$ function of this layer accepts the input feature tensor and the generated convolution triplets (typically pre-sorted by the kernel index $k$) to execute the operation. This design ensures that the GPU pipelines remain saturated with dense arithmetic (Matrix-Vector Multiplication) rather than stalling on irregular memory access.

It is worth emphasizing that in this framework, **feature tensors are first-class citizens**. They are the primary carriers of the learned signal and the subject of all gradient backpropagation. The spatial coordinates, having served their purpose in generating the neighbor lists and triplets, are treated simply as "metadata" that guides the data-weight flow, rather than being part of the arithmetic computation itself.

