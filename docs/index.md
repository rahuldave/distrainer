# distrainer

Block-native distributed training on Ray Train: a corpus cut into immutable blocks, dealt to
ranks from an append-only log, with a four-integer ledger in every checkpoint so a run resumes
row-exactly at any world size, on a laptop, on containers, on cloud machines or on GPU pods.
The code is on [GitHub](https://github.com/rahuldave/distrainer).

## Start here

- [An introduction to distributed training with Ray, and how distrainer does it](introduction.md)
- [Ways to run distrainer](running-modes.md): laptop, OrbStack containers, Kubernetes, uncloud machines, RunPod pods
- [Command-line reference](cli.md)

## Tutorials

1. [Batch training on a block log](tutorials/batch.md)
2. [Streaming, hooks and retention](tutorials/streaming.md)
3. [The same run on Kubernetes (KubeRay)](tutorials/kuberay.md)
4. [The same run on a cluster of machines (uncloud)](tutorials/uncloud.md)
5. [The cluster on AWS, an arm64 bed and an x86 bed, S3, a private registry](tutorials/aws.md)
6. [An image contrastive example, its GPU image, and RunPod pods](tutorials/runpod.md)
7. [The parallel kinds: DDP, local SGD, DiLoCo, FSDP on the same log](tutorials/parallel.md)

## Concepts

- [Collectives: the primitives every distributed training is built from](collectives.md)
- [Parallel training, in terms of the collectives, and where distrainer fits](parallelism.md)
- [Retention and garbage collection](retention.md)
- [Examples and verification scenarios](examples-and-scenarios.md): how every scenario is driven and checked

## Reference

- [The specification (v0.1)](distrainer-spec.md): interfaces, the loop, checkpoints, configuration, the harness, the milestones
- [The design sketch](distrainer-design.md) that preceded the spec
- [Research: training between batch and epoch granularity in Ray and Anyscale](ray-sub-epoch-training-report.md)

## Field notes

- [RunPod gotchas](runpod-gotchas.md)
- [uncloud gotchas](uncloud-gotchas.md)
